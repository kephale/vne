from typing import Tuple

import numpy as np
import torch


class ShapeSimilarityLoss:
    """Shape similarity loss based on pre-calculated shape similarity.

    Parameters
    ----------
    lookup : torch.Tensor (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The shape similarity should be normalized to the range (-1, 1).

    Notes
    -----
    The final loss is calculated using L1-norm. This could be changed, e.g.
    L2-norm. Not sure what the best one is yet.
    """

    def __init__(self, lookup: torch.Tensor):
        self.lookup = lookup
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = torch.nn.L1Loss()

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Return the shape similarity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the objects. These indices should correspond to the
            rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.

        Returns
        -------
        loss : torch.Tensor
            The shape similarity loss.
        """
        # first calculate the shape similarity for the real classes
        c = torch.combinations(y_true, r=2, with_replacement=False)
        shape_similarity = self.lookup[c[:, 0], c[:, 1]]

        # now calculate the latent similarity
        z_id = torch.tensor(list(range(y_pred.shape[0])))
        c = torch.combinations(z_id, r=2, with_replacement=False)
        latent_similarity = self.cos(y_pred[c[:, 0], :], y_pred[c[:, 1], :])

        loss = self.l1loss(latent_similarity, shape_similarity)
        return loss


class AffinityCosineLoss:
    """Affinity loss based on pre-calculated shape similarity.

    Parameters
    ----------
    lookup : np.ndarray (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The affinity similarity should be normalized to the range
        (-1, 1).

    Notes
    -----
    The final loss is calculated using L1-norm. This could be changed, e.g.
    L2-norm. Not sure what the best one is yet.
    """

    def __init__(self, lookup: torch.Tensor, device: torch.device):
        self.device = device
        self.lookup = torch.tensor(lookup).to(device)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        self.l1loss = torch.nn.L1Loss(reduction="none")  # Use "none" for per-sample loss

    def __call__(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, per_sample: bool = False
    ) -> torch.Tensor:
        """Return the affinity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the object as an index. These indices should
            correspond to the rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.
        per_sample : bool, optional
            If True, return the per-sample loss for each pair in the batch. If False,
            return the mean loss for the entire batch.

        Returns
        -------
        loss : torch.Tensor
            The affinity loss. If `per_sample` is True, returns a tensor of
            per-sample losses for each pair. Otherwise, returns the mean loss.
        """
        # Calculate affinity from the lookup table for real classes
        c = (
            torch.combinations(y_true, r=2, with_replacement=False)
            .to(self.device)
            .long()
        )
        affinity = self.lookup[c[:, 0], c[:, 1]].to(self.device)

        # Calculate latent similarity
        z_id = torch.arange(y_pred.shape[0], device=self.device)
        c_latent = torch.combinations(z_id, r=2, with_replacement=False)
        latent_similarity = self.cos(y_pred[c_latent[:, 0], :], y_pred[c_latent[:, 1], :])

        # Calculate L1 loss
        losses = self.l1loss(latent_similarity, affinity)

        if per_sample:
            # Return per-sample losses
            return losses
        else:
            # Return mean loss
            return torch.mean(losses)


class AffinityVAE(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        *,
        latent_dims: int = 8,
        pose_channels: int = 1,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.pose_channels = pose_channels

        flat_shape = self.encoder.flat_shape

        self.mu = torch.nn.Linear(flat_shape, latent_dims)
        self.log_var = torch.nn.Linear(flat_shape, latent_dims)
        self.pose = torch.nn.Linear(flat_shape, pose_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mu, log_var, pose = self.encode(x)
        z = self.reparameterise(mu, log_var)
        x = self.decode(z, pose)
        return x, z, pose, mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        return mu, log_var, pose

    def decode(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, pose)


class ShapeVAE(torch.nn.Module):
    """Shape regularized variational autoencoder.

    Parameters
    ----------
    latent_dims : int
        The size of the latent representation.
    pose_dims : int
        The size of the pose representation.
    spatial_dims : int (2 or 3)
        Planar of volumetric data.

    """

    def __init__(
        self, latent_dims: int = 8, pose_dims: int = 1, spatial_dims: int = 2
    ):
        super(ShapeVAE, self).__init__()

        if spatial_dims == 2:
            conv = torch.nn.Conv2d
            conv_T = torch.nn.ConvTranspose2d
            unflat_shape = (64, 4, 4)
        elif spatial_dims == 3:
            conv = torch.nn.Conv3d
            conv_T = torch.nn.ConvTranspose3d
            unflat_shape = (64, 4, 4, 4)
        else:
            raise ValueError(
                f"`spatial_dims` must be in (2, 3), got: {spatial_dims}."
            )

        flat_shape = np.prod(unflat_shape)

        self.encoder = torch.nn.Sequential(
            conv(1, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            conv(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            conv(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            conv(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dims + pose_dims, flat_shape),
            torch.nn.Unflatten(-1, unflat_shape),
            conv_T(64, 32, 3, stride=2),
            torch.nn.ReLU(True),
            conv_T(32, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            conv_T(16, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(True),
            conv_T(8, 1, 2, stride=2, padding=1),
        )

        self.mu = torch.nn.Linear(flat_shape, latent_dims)
        self.log_var = torch.nn.Linear(flat_shape, latent_dims)
        self.pose = torch.nn.Linear(flat_shape, pose_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        mu, log_var, pose = self.encode(x)
        z = self.reparameterise(mu, log_var)
        z_pose = torch.cat([pose, z], dim=-1)
        x = self.decode(z_pose)
        return x, z, z_pose, mu, log_var

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        pose = self.pose(encoded)
        return mu, log_var, pose

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops.

    Parameters
    ----------
    start: int
        The size of the layer before pooling.
    n_pools: int
        The number of pooling operations.

    Returns
    -------
    dims: int
        The size of the layer after pooling.
    """
    return start // (2**n_pools)
