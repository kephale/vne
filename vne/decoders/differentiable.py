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

        # Create coordinate grid and register as buffer for proper device management
        grids = torch.meshgrid(
            *[torch.linspace(-1, 1, sz) for sz in shape],
            indexing="xy",
        )

        # add all zeros for z- if we have a 2d grid
        if len(shape) == SpatialDims.TWO:
            grids += (torch.zeros_like(grids[0]),)

        coords = torch.stack([torch.ravel(grid) for grid in grids], axis=0).transpose(0, 1).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(
        self,
        splats: torch.Tensor, 
        weights: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        splat_sigma_range: Tuple[float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Render the Gaussian splats with correct tensor dimensions."""
        
        # Ensure all inputs are on the same device as coords
        device = self.coords.device
        splats = splats.to(device)
        weights = weights.to(device)
        sigmas = sigmas.to(device)
        
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

        self._device = device
        self._shape = shape
        self._ndim = len(shape)
        self._output_channels = output_channels
        self._splat_sigma_range = splat_sigma_range
        self._default_axis = default_axis.as_tensor()

        # Register networks and move to specified device
        self.centroids = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_splats * 3, n_splats * 3),
            torch.nn.Tanh(),
        ).to(device)

        self.weights = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Tanh(),
            SoftStep(k=10.0),
        ).to(device)

        self.sigmas = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Sigmoid(),
        ).to(device)

        # Initialize renderer
        self._splatter = GaussianSplatRenderer(
            shape,
            device=device,
        ).to(device)

        # Add final conv decoder if needed
        if output_channels is not None:
            conv = (
                torch.nn.Conv3d
                if self._ndim == SpatialDims.THREE
                else torch.nn.Conv2d
            )

            self._decoder = torch.nn.Sequential(
                Negate(),
                conv(1, 1, kernel_size=1),
                conv(1, output_channels, kernel_size=9, padding="same"),
            ).to(device)

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

        # Move inputs to correct device and predict parameters
        z = z.to(self._device)
        pose = pose.to(self._device)
        
        splats = self.centroids(z).view(z.shape[0], 3, -1)
        weights = self.weights(z)
        sigmas = self.sigmas(z)

        # Get batch size
        batch_size = z.shape[0]

        # Handle single dimension pose
        if pose.shape[-1] == 1:
            pose = torch.concat(
                [
                    pose,
                    torch.tile(self._default_axis, (batch_size, 1)).to(self._device),
                ],
                axis=-1,
            )

        # Convert axis angles to quaternions
        assert pose.shape[-1] == 4, pose.shape
        quaternions = axis_angle_to_quaternion(pose, normalize=True)

        # Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # Rotate the 3D points using the rotation matrices
        rotated_splats = torch.matmul(
            rotation_matrices,
            splats,
        )

        # Use only the required spatial dimensions
        rotated_splats = rotated_splats[:, :self._ndim, :]

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

        # Decode the splats from the latents and pose
        splats, weights, sigmas = self.decode_splats(z, pose)

        # Apply the gaussian splat renderer
        x = self._splatter(
            splats, weights, sigmas, splat_sigma_range=self._splat_sigma_range
        )

        # Apply final convolution if needed
        if self._output_channels is not None and use_final_convolution:
            x = self._decoder(x)

        return x


class DownsampledGaussianSplatRenderer(BaseDecoder):
    """Perform gaussian splatting at a lower resolution."""

    def __init__(
        self,
        target_shape: tuple,
        downsample_factor: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self._target_shape = target_shape
        self._ndim = len(target_shape)
        self.downsample_factor = downsample_factor
        
        # Calculate downsampled shape
        self._internal_shape = tuple(s // downsample_factor for s in target_shape)

        if len(target_shape) not in (SpatialDims.TWO, SpatialDims.THREE):
            raise ValueError("Only 2D or 3D rotations are currently supported")

        # Create coordinate grid at lower resolution
        grids = torch.meshgrid(
            *[torch.linspace(-1, 1, sz) for sz in self._internal_shape],
            indexing="xy",
        )

        # Add zeros for z if we have a 2d grid
        if len(target_shape) == SpatialDims.TWO:
            grids += (torch.zeros_like(grids[0]),)

        coords = torch.stack([torch.ravel(grid) for grid in grids], axis=0).transpose(0, 1).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(
        self,
        splats: torch.Tensor, 
        weights: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        splat_sigma_range: tuple = (0.0, 1.0),
    ) -> torch.Tensor:
        """Render the Gaussian splats at lower resolution and upsample."""
        
        # Ensure all inputs are on the same device as coords
        device = self.coords.device
        splats = splats.to(device)
        weights = weights.to(device)
        sigmas = sigmas.to(device)
        
        # Adjust sigmas for downsampled space
        sigmas = sigmas * self.downsample_factor
        
        # Clamp weights
        weights = torch.clamp(weights, 0.0, 1.0)
        
        # Scale sigma values with clamping
        min_sigma, max_sigma = splat_sigma_range
        sigmas = torch.clamp(
            sigmas * (max_sigma - min_sigma) + min_sigma,
            min=1e-6,
            max=1.0
        )

        # Transpose splats for computation
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

        # Reshape to internal resolution
        x = x.reshape((-1, *self._internal_shape)).unsqueeze(1)
        
        # Upsample to target resolution using trilinear interpolation
        if self._ndim == 3:
            x = F.interpolate(x, size=self._target_shape, mode='trilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=self._target_shape, mode='bilinear', align_corners=False)

        return x

class DownsampledGaussianSplatDecoder(BaseDecoder):
    """Memory-efficient Gaussian splat decoder that operates at lower internal resolution.

    Parameters
    ----------
    shape : tuple
        Target output shape of the image data (final resolution)
    downsample_factor : int
        Factor by which to downsample internal computations (e.g., 2 means operate at half resolution)
    n_splats : int
        The number of Gaussians in the mixture model
    latent_dims : int
        The dimensions of the latent representation
    output_channels : int, optional
        The number of output channels in the final image volume
    splat_sigma_range : tuple[float]
        The minimum and maximum sigma values for each splat
    default_axis : CartesianAxes
        Default cartesian axis for rotation
    """

    def __init__(
        self,
        shape: tuple,
        downsample_factor: int = 2,
        n_splats: int = 128,
        latent_dims: int = 8,
        output_channels: int = None,
        splat_sigma_range: tuple = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self._device = device
        self._shape = shape
        self._ndim = len(shape)
        self._output_channels = output_channels
        self._splat_sigma_range = splat_sigma_range
        self._default_axis = default_axis.as_tensor()
        self.downsample_factor = downsample_factor

        # Networks for predicting Gaussian parameters
        self.centroids = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(n_splats * 3, n_splats * 3),
            torch.nn.Tanh(),
        ).to(device)

        self.weights = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Tanh(),
            torch.nn.Sigmoid(),
        ).to(device)

        self.sigmas = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Sigmoid(),
        ).to(device)

        # Initialize renderer with downsampling
        self._splatter = DownsampledGaussianSplatRenderer(
            shape,
            downsample_factor=downsample_factor,
            device=device,
        ).to(device)

        # Add final conv decoder if needed
        if output_channels is not None:
            conv = (
                torch.nn.Conv3d
                if self._ndim == SpatialDims.THREE
                else torch.nn.Conv2d
            )

            # Define upsampling layer
            if self._ndim == SpatialDims.THREE:
                self._decoder = torch.nn.Sequential(
                    conv(1, 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    conv(32, output_channels, kernel_size=3, padding=1),
                ).to(device)

    def decode_splats(
        self, z: torch.Tensor, pose: torch.Tensor
    ) -> tuple[torch.Tensor]:
        """Decode the splats to retrieve the coordinates, weights and sigmas."""
        if pose.shape[-1] not in (1, 4):
            raise ValueError(
                "Pose needs to be either a single angle rotation about the "
                "`default_axis` or a full angle-axis representation in 3D. "
            )

        # Move inputs to correct device and predict parameters
        z = z.to(self._device)
        pose = pose.to(self._device)
        
        splats = self.centroids(z).view(z.shape[0], 3, -1)
        weights = self.weights(z)
        sigmas = self.sigmas(z)

        # Get batch size
        batch_size = z.shape[0]

        # Handle single dimension pose
        if pose.shape[-1] == 1:
            pose = torch.concat(
                [
                    pose,
                    torch.tile(self._default_axis, (batch_size, 1)).to(self._device),
                ],
                axis=-1,
            )

        # Convert axis angles to quaternions
        quaternions = axis_angle_to_quaternion(pose, normalize=True)

        # Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # Rotate the 3D points using the rotation matrices
        rotated_splats = torch.matmul(
            rotation_matrices,
            splats,
        )

        # Use only the required spatial dimensions
        rotated_splats = rotated_splats[:, :self._ndim, :]

        return rotated_splats, weights, sigmas

    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        *,
        use_final_convolution: bool = True,
    ) -> torch.Tensor:
        """Decode the latents to an image volume given an explicit transform."""

        # Decode the splats from the latents and pose
        splats, weights, sigmas = self.decode_splats(z, pose)

        # Apply the gaussian splat renderer (operates at lower resolution internally)
        x = self._splatter(
            splats, weights, sigmas, splat_sigma_range=self._splat_sigma_range
        )

        # Apply final convolution if needed
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
        curriculum_schedule = None,
        splat_sigma_range: Tuple[float, float] = (0.00025, 0.2),
        default_axis: CartesianAxes = CartesianAxes.Z,  # Add default axis parameter
    ):
        super().__init__()
        
        self._shape = shape
        self._device = device
        self._ndim = len(shape)
        self._output_channels = output_channels
        self._n_gaussians_range = n_gaussians_range
        self._d_model = d_model
        self.curriculum_schedule = curriculum_schedule
        self.current_epoch = 0
        self.splat_sigma_range = splat_sigma_range
        self._default_axis = default_axis.as_tensor()  # Store default axis

        # Initialize sequence length predictor
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

    def decode_sequence(self, z: torch.Tensor, pose: torch.Tensor):
        """Decode latent vector into sequence of Gaussian parameters with pose transformation."""
        batch_size = z.shape[0]
        
        # Use curriculum-adjusted maximum sequence length
        min_len, max_len = self._n_gaussians_range
        
        # Predict sequence length
        seq_len_ratio = self.sequence_length(z).squeeze(-1)
        n_gaussians = min_len + (max_len - min_len) * seq_len_ratio
        n_gaussians = n_gaussians.round().long()
        
        # Use curriculum-adjusted max_len for all sequences
        query_embeddings = self.query_embed.weight[:max_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Project latent vector
        memory = self.latent_projection(z).unsqueeze(1)
        
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
        
        # Scale sigmas to desired range
        min_sigma, max_sigma = self.splat_sigma_range
        raw_sigmas = params[..., self._ndim+1:].sigmoid()
        sigmas = min_sigma + (max_sigma - min_sigma) * raw_sigmas
        
        # Create mask for valid positions
        batch_indices = torch.arange(batch_size, device=z.device)
        position_mask = torch.arange(max_len, device=z.device).unsqueeze(0) < n_gaussians.unsqueeze(1)
        
        # Apply mask
        positions = positions * position_mask.unsqueeze(-1)
        amplitudes = amplitudes * position_mask
        sigmas = sigmas * position_mask.unsqueeze(-1)
        
        # Handle pose transformation
        if pose.shape[-1] not in (1, 4):
            raise ValueError(
                "Pose needs to be either a single angle rotation about the "
                "`default_axis` or a full angle-axis representation in 3D."
            )

        # Handle single dimension pose
        if pose.shape[-1] == 1:
            pose = torch.concat(
                [
                    pose,
                    torch.tile(self._default_axis, (batch_size, 1)).to(self._device),
                ],
                axis=-1,
            )

        # Convert axis angles to quaternions
        quaternions = axis_angle_to_quaternion(pose, normalize=True)

        # Convert quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # Add third spatial dimension if needed for rotation
        if positions.shape[-1] == 2:
            zeros = torch.zeros_like(positions[..., :1])
            positions = torch.cat([positions, zeros], dim=-1)

        # Rotate the positions
        positions = torch.einsum('bij,bkj->bki', rotation_matrices, positions)

        # Keep only required spatial dimensions
        positions = positions[..., :self._ndim]
        
        return positions, amplitudes, sigmas

    def decode_splats(self, z: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor]:
        """Decode the splats to retrieve the coordinates, weights and sigmas."""
        # Get sequence of Gaussian parameters with pose transformation
        positions, amplitudes, sigmas = self.decode_sequence(z, pose)
        
        # Reshape outputs for compatibility
        batch_size = positions.shape[0]
        n_gaussians = positions.shape[1]
        
        # Reshape positions to match expected format (batch, 3, n_gaussians)
        rotated_splats = positions.transpose(1, 2)
        
        # Add third spatial dimension if needed
        if rotated_splats.shape[1] == 2:
            zeros = torch.zeros(batch_size, 1, n_gaussians, device=z.device)
            rotated_splats = torch.cat([rotated_splats, zeros], dim=1)
        
        # Convert amplitudes and sigmas to expected format
        weights = amplitudes
        
        # Ensure sigmas has correct shape (batch, n_gaussians)
        if len(sigmas.shape) == 3:
            sigmas = torch.mean(sigmas, dim=2)
        
        return rotated_splats, weights, sigmas

    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        *,
        use_final_convolution: bool = True,
    ) -> torch.Tensor:
        # Get sequence of Gaussian parameters with pose transformation
        positions, amplitudes, sigmas = self.decode_sequence(z, pose)
        
        # Generate Gaussian grid
        x = self._generate_gaussian_grid(positions, amplitudes, sigmas)
        
        # Optional final convolution
        if self._output_channels is not None and use_final_convolution:
            x = self.final_conv(x)
            
        return x

    def update_epoch(self, epoch):
        """Update the current epoch for curriculum learning."""
        self.current_epoch = epoch
        if self.curriculum_schedule:
            max_gaussians = self.curriculum_schedule.get_max_gaussians(epoch)
            self._n_gaussians_range = (self._n_gaussians_range[0], max_gaussians)

    def _generate_gaussian_grid(self, positions, amplitudes, sigmas):
        """Generate grid of Gaussian values using accumulation approach."""
        device = positions.device
        batch_size = positions.shape[0]
        
        # Initialize output grid
        result = torch.zeros(batch_size, 1, *self._shape, device=device)
        
        # Create coordinate grids once and expand for broadcasting
        x_grid = torch.linspace(-1, 1, self._shape[0], device=device)
        y_grid = torch.linspace(-1, 1, self._shape[1], device=device)
        z_grid = torch.linspace(-1, 1, self._shape[2], device=device)
        
        # Reshape parameters
        positions = positions.view(batch_size, -1, positions.shape[-1])  # [batch, N, 3]
        sigmas = sigmas.view(batch_size, -1, sigmas.shape[-1])  # [batch, N, 3]
        amplitudes = amplitudes.view(batch_size, -1)  # [batch, N]
        
        # Process gaussians in chunks to reduce memory usage
        chunk_size = 16  # Reduced chunk size for better memory management
        num_gaussians = positions.shape[1]
        
        for chunk_start in range(0, num_gaussians, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_gaussians)
            
            # Get current chunk parameters
            pos_chunk = positions[:, chunk_start:chunk_end, :]  # [batch, chunk, 3]
            sig_chunk = sigmas[:, chunk_start:chunk_end, :]    # [batch, chunk, 3]
            amp_chunk = amplitudes[:, chunk_start:chunk_end]    # [batch, chunk]
            
            for b in range(batch_size):
                for i in range(chunk_end - chunk_start):
                    # Calculate 1D gaussian components
                    x_diff = (x_grid - pos_chunk[b, i, 0]) / (sig_chunk[b, i, 0] + 1e-6)
                    y_diff = (y_grid - pos_chunk[b, i, 1]) / (sig_chunk[b, i, 1] + 1e-6)
                    z_diff = (z_grid - pos_chunk[b, i, 2]) / (sig_chunk[b, i, 2] + 1e-6)
                    
                    # Compute 1D gaussians
                    gaussian_x = torch.exp(-0.5 * x_diff * x_diff)  # [X]
                    gaussian_y = torch.exp(-0.5 * y_diff * y_diff)  # [Y]
                    gaussian_z = torch.exp(-0.5 * z_diff * z_diff)  # [Z]
                    
                    # Compute 3D gaussian using proper broadcasting
                    # Reshape for correct broadcasting: [X, 1, 1] * [1, Y, 1] * [1, 1, Z]
                    gaussian_3d = (gaussian_x.view(-1, 1, 1) * 
                                gaussian_y.view(1, -1, 1) * 
                                gaussian_z.view(1, 1, -1))
                    
                    # Scale by amplitude and accumulate
                    result[b, 0] += amp_chunk[b, i] * gaussian_3d
                    
                    # Free memory explicitly
                    del gaussian_x, gaussian_y, gaussian_z, gaussian_3d
            
        return torch.clamp(result, 0.0, 1.0)
