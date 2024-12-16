import torch
from torch import nn

import numpy as np

from vne.decoders.base import BaseDecoder
from vne.decoders.spatial import (
    CartesianAxes,
    SpatialDims,
    axis_angle_to_quaternion,
    quaternion_to_rotation_matrix,
)

from typing import Optional, Tuple

# From https://github.com/alan-turing-institute/affinity-vae/blob/gsd_binarised_weights_single_conv_lager/avae/decoders/differentiable.py
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)


class StraightThroughEstimator(torch.nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class GaussianSplatRenderer(BaseDecoder):
    """Perform gaussian splatting."""

    def __init__(
        self,
        shape: Tuple[int],
        *,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self._shape = shape
        self._ndim = len(shape)

        if len(shape) not in (SpatialDims.TWO, SpatialDims.THREE):
            raise ValueError("Only 2D or 3D rotations are currently supported")

        grids = torch.meshgrid(
            *[torch.linspace(-1, 1, sz, device=device) for sz in shape],
            indexing="xy",
        )

        # add all zeros for z- if we have a 2d grid
        if len(shape) == SpatialDims.TWO:
            grids += (
                torch.zeros_like(
                    grids[0],
                ),
            )

        self.coords = (
            torch.stack([torch.ravel(grid) for grid in grids], axis=0)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(device)
        )

    def forward(
        self,
        splats: torch.Tensor, 
        weights: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        splat_sigma_range: Tuple[float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Render the Gaussian splats with correct tensor dimensions."""
        
        # Clamp weights to prevent explosion
        weights = torch.clamp(weights, 0.0, 1.0)
        
        # Scale the sigma values with clamping
        min_sigma, max_sigma = splat_sigma_range
        sigmas = torch.clamp(
            sigmas * (max_sigma - min_sigma) + min_sigma,
            min=1e-6,
            max=1.0
        )

        # Transpose splats for efficient computation
        splats_t = splats.transpose(1, 2)  # [B, N, D]

        # Calculate squared distances efficiently
        coords_norm = torch.sum(self.coords ** 2, dim=-1, keepdim=True)  # [B, M, 1]
        splats_norm = torch.sum(splats_t ** 2, dim=-1)  # [B, N]
        
        # Compute cross term
        cross_term = torch.matmul(self.coords, splats)  # [B, M, N]
        
        # Combine terms for full distance calculation
        D_squared = coords_norm + splats_norm.unsqueeze(1) - 2 * cross_term
        D_squared = torch.clamp(D_squared, min=0.0)

        # Scale gaussians with numerical stability
        sigmas = 2.0 * sigmas.unsqueeze(1) ** 2  # [B, 1, N]
        
        # Calculate gaussian values with stability checks
        gaussian_values = weights.unsqueeze(1) * torch.exp(
            torch.clamp(-D_squared / sigmas, min=-88.0)
        )
        
        # Sum and normalize
        x = torch.sum(gaussian_values, dim=-1)
        x = torch.clamp(x, 0.0, 1.0)

        return x.reshape((-1, *self._shape)).unsqueeze(1)

class SoftStep(torch.nn.Module):
    """Soft (differentiable) step function in the range of 0-1."""

    def __init__(self, *, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.k * x))

class Negate(torch.nn.Module):
    def forward(self, x):
        return -x


class GaussianSplatDecoder(BaseDecoder):
    """Differentiable Gaussian splat decoder.

    Parameters
    ----------
    shape : tuple
        A tuple describing the output shape of the image data. Can be 2- or 3-
        dimensional. For example: (32, 32, 32)
    n_splats : int
        The number of Gaussians in the mixture model.
    latent_dims : int
        The dimensions of the latent representation.
    output_channels : int, optional
        The number of output channels in the final image volume. If not
        supplied, this will default to 1. If it is supplied, additional
        convolutions are applied to the GMM model.
    splat_sigma_range : tuple[float]
        The minimum and maximum sigma values for each splat. Useful to control
        the resolution of the final render.
    default_axis : CartesianAxes
        A default cartesian axis to use for rotation if the pose is provided by
        a rotation only. Default is Z, equivalent to a typical image rotation
        about the central axis.

    Notes
    -----
    Takes the latent code and pose estimate to generate a planar or volumetric
    image.  The code is used to position N symmetric gaussians in the image
    volume which are then rotated by an explicit rotation transform. These are
    rendered as an image by evaluating the list of gaussians as a GMM.

    The renderer is differentiable and can therefore be used during training.
    """

    def __init__(
        self,
        shape: Tuple[int],
        *,
        n_splats: int = 128,
        latent_dims: int = 8,
        output_channels: Optional[int] = None,
        splat_sigma_range: Tuple[float, float] = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        # centroids should be in the range of (-1, 1)
        self.centroids = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_splats * 3, n_splats * 3),
            torch.nn.Tanh(),
        )

        # weights are effectively whether a splat is used or not
        # use a soft step function to make this `binary` (but differentiable)
        # NOTE(arl): not sure if this really makes any difference
        self.weights = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Tanh(),
            SoftStep(k=10.0),
            # StraightThroughEstimator(),
        )

        # sigma ends up being scaled by `splat_sigma_range`
        self.sigmas = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Sigmoid(),
        )

        # now set up the differentiable renderer
        self.configure_renderer(
            shape,
            splat_sigma_range=splat_sigma_range,
            default_axis=default_axis,
            device=device,
        )

        self._device = device
        self._ndim = len(shape)
        self._output_channels = output_channels

        # add a final convolutional decoder to generate an image if the number
        # of output channels has been provided
        if output_channels is not None:
            conv = (
                torch.nn.Conv3d
                if self._ndim == SpatialDims.THREE
                else torch.nn.Conv2d
            )

            # New final convolutional decoder pipeline
            self._decoder = torch.nn.Sequential(
                Negate(),  # Negate the density (electron density)
                conv(1, 1, kernel_size=1),  # Scaling and offset (1x1x1 convolution)
                conv(1, output_channels, kernel_size=9, padding="same"),  # Learn the CTF
            )

    def configure_renderer(
        self,
        shape: Tuple[int],
        *,
        splat_sigma_range: Tuple[float, float] = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Reconfigure the renderer.

        Notes
        -----
        This might be useful to do once a model is trained. For example, one
        could change the resolution of the rendered image by changing the
        `shape` of the output.
        """
        self._shape = shape
        self._default_axis = default_axis.as_tensor()
        self._splatter = GaussianSplatRenderer(
            shape,
            device=device,
        )
        self._splat_sigma_range = splat_sigma_range

    def decode_splats(
        self, z: torch.Tensor, pose: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Decode the splats to retrieve the coordinates, weights and sigmas."""
        if pose.shape[-1] not in (1, 4):
            raise ValueError(
                "Pose needs to be either a single angle rotation about the "
                "`default_axis` or a full angle-axis representation in 3D. "
            )

        # predict the centroids for the splats
        z = z.to(self._device)
        pose = pose.to(self._device)
        splats = self.centroids(z).view(z.shape[0], 3, -1).to(self._device)
        weights = self.weights(z).to(self._device)
        sigmas = self.sigmas(z).to(self._device)


        # get the batch size
        batch_size = z.shape[0]

        # in the case where the encoded pose only has one dimension, we need to
        # use the pose as a rotation about the z-axis
        if pose.shape[-1] == 1:
            pose = torch.concat(
                [
                    pose,
                    torch.tile(self._default_axis, (batch_size, 1)),
                ],
                axis=-1,
            )

        # convert axis angles to quaternions
        assert pose.shape[-1] == 4, pose.shape
        quaternions = axis_angle_to_quaternion(pose, normalize=True)

        # convert the quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions).to(self._device)

        # rotate the 3D points using the rotation matrices
        rotated_splats = torch.matmul(
            rotation_matrices,
            splats,
        ).to(self._device)

        # use only the required spatial dimensions (batch, ndim, samples)
        rotated_splats = rotated_splats[:, : self._ndim, :]

        return rotated_splats, weights, sigmas

    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        *,
        use_final_convolution: bool = True,
    ) -> torch.Tensor:
        """Decode the latents to an image volume given an explicit transform.

        Parameters
        ----------
        z : tensor
            An (N, D) tensor specifying the D dimensional latent encodings for
            the minibatch of N images.
        pose : tensor
            An (N, 1 | 4) tensor specifying the pose in terms of a single
            rotation (assumed around the z-axis) or a full axis-angle rotation.
        use_final_convolution: bool
            Whether to apply the final convolutional layers to recover the image.
            This can be useful to inspect the underlying structure in a trained
            model.

        Returns
        -------
        x : tensor
            The decoded image from the latents and pose.
        """

        # decode the splats from the latents and pose
        splats, weights, sigmas = self.decode_splats(z, pose)

        x = self._splatter(
            splats, weights, sigmas, splat_sigma_range=self._splat_sigma_range
        )

        # if we're doing a final convolution, do it here
        if self._output_channels is not None and use_final_convolution:
            x = self._decoder(x)

        return x

class TransformerGaussianDecoder(BaseDecoder):
    def __init__(
        self,
        shape: Tuple[int],
        n_gaussians_range: Tuple[int, int] = (16, 128),
        latent_dims: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        output_channels: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        
        self._shape = shape
        self._device = device
        self._ndim = len(shape)
        self._output_channels = output_channels
        self._n_gaussians_range = n_gaussians_range
        self._d_model = d_model

        # Sequence length predictor
        self.sequence_length = nn.Sequential(
            nn.Linear(latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Project latent vector to transformer dimension
        self.latent_projection = nn.Linear(latent_dims, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Learned query embeddings
        self.query_embed = nn.Embedding(n_gaussians_range[1], d_model)
        
        # Project transformer outputs to Gaussian parameters
        self.gaussian_params = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 7)  # x,y,z position + amplitude + sigma_x,y,z
        )
        
        if output_channels is not None:
            conv = nn.Conv3d if self._ndim == 3 else nn.Conv2d
            self.final_conv = nn.Sequential(
                conv(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                conv(16, output_channels, kernel_size=1)
            )

    def _generate_gaussian_grid(self, positions, amplitudes, sigmas):
        """Generate grid of Gaussian values."""
        device = positions.device
        batch_size = positions.shape[0]

        # Create coordinate grid
        coords = [torch.linspace(-1, 1, s, device=device) for s in self._shape]
        grid_points = torch.meshgrid(*coords, indexing='ij')
        grid_coords = torch.stack([g.reshape(-1) for g in grid_points], dim=-1)
        
        # Reshape positions and parameters for broadcasting
        positions = positions.view(batch_size, -1, positions.shape[-1])  # [batch, N, 3]
        sigmas = sigmas.view(batch_size, -1, sigmas.shape[-1])  # [batch, N, 3]
        amplitudes = amplitudes.view(batch_size, -1)  # [batch, N]
        
        # Calculate distances efficiently
        grid_coords = grid_coords.unsqueeze(0).unsqueeze(2)  # [1, P, 1, 3]
        positions = positions.unsqueeze(1)  # [batch, 1, N, 3]
        sigmas = sigmas.unsqueeze(1)  # [batch, 1, N, 3]
        amplitudes = amplitudes.unsqueeze(1)  # [batch, 1, N]
        
        # Calculate squared distances
        diff = (grid_coords - positions) / (sigmas + 1e-6)
        dist_sq = torch.sum(diff * diff, dim=-1)
        
        # Calculate Gaussian values
        gaussians = amplitudes * torch.exp(-0.5 * dist_sq)
        
        # Sum over Gaussians
        result = torch.sum(gaussians, dim=-1)
        
        # Reshape to spatial dimensions
        result = result.view(batch_size, 1, *self._shape)
        
        return result

    def decode_sequence(self, z: torch.Tensor):
        """Decode latent vector into sequence of Gaussian parameters."""
        batch_size = z.shape[0]
        
        # Predict sequence length - modify to ensure valid range
        seq_len_ratio = self.sequence_length(z).squeeze(-1)  # Shape: [batch_size]
        min_len, max_len = self._n_gaussians_range
        
        # Calculate number of gaussians for each batch element
        n_gaussians = min_len + (max_len - min_len) * seq_len_ratio
        n_gaussians = n_gaussians.round().long()  # Convert to integer
        
        # Use max_len for all sequences to avoid inconsistent sizes
        # We'll mask unused positions later
        query_embeddings = self.query_embed.weight[:max_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Project latent vector
        memory = self.latent_projection(z).unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Generate sequence with transformer
        transformer_out = self.transformer_decoder(
            query_embeddings,
            memory
        )
        
        # Convert to Gaussian parameters
        params = self.gaussian_params(transformer_out)
        
        # Split parameters
        positions = params[..., :self._ndim].tanh()  # Positions in [-1,1]
        amplitudes = params[..., self._ndim].sigmoid()  # Amplitudes in [0,1]
        sigmas = 0.1 + 0.9 * params[..., self._ndim+1:].sigmoid()  # Sigmas in [0.1,1]
        
        # Create mask for valid positions
        batch_indices = torch.arange(batch_size, device=z.device)
        position_mask = torch.arange(max_len, device=z.device).unsqueeze(0) < n_gaussians.unsqueeze(1)
        
        # Apply mask
        positions = positions * position_mask.unsqueeze(-1)
        amplitudes = amplitudes * position_mask
        sigmas = sigmas * position_mask.unsqueeze(-1)
        
        return positions, amplitudes, sigmas

    def decode_splats(self, z: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor]:
        """Decode the splats to retrieve the coordinates, weights and sigmas."""
        # Get sequence of Gaussian parameters
        positions, amplitudes, sigmas = self.decode_sequence(z)
        
        # Reshape outputs for compatibility with other decoders
        batch_size = positions.shape[0]
        n_gaussians = positions.shape[1]
        
        # Reshape positions to match expected format (batch, 3, n_gaussians)
        rotated_splats = positions.transpose(1, 2)
        
        # Add third spatial dimension if needed
        if rotated_splats.shape[1] == 2:
            zeros = torch.zeros(batch_size, 1, n_gaussians, device=z.device)
            rotated_splats = torch.cat([rotated_splats, zeros], dim=1)
        
        # Convert amplitudes and sigmas to expected format
        weights = amplitudes  # Already in correct shape (batch, n_gaussians)
        
        # Ensure sigmas has correct shape (batch, n_gaussians)
        if len(sigmas.shape) == 3:  # If we have different sigma per dimension
            sigmas = torch.mean(sigmas, dim=2)  # Average across dimensions
        
        return rotated_splats, weights, sigmas

    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        *,
        use_final_convolution: bool = True,
    ) -> torch.Tensor:
        # Get sequence of Gaussian parameters
        positions, amplitudes, sigmas = self.decode_sequence(z)
        
        # Generate Gaussian grid
        x = self._generate_gaussian_grid(positions, amplitudes, sigmas)
        
        # Optional final convolution
        if self._output_channels is not None and use_final_convolution:
            x = self.final_conv(x)
            
        return x