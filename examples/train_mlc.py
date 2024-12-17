import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import requests
import torch.multiprocessing as mp
import time

if __name__ == '__main__':
    mp.set_start_method('spawn')

from torchvision.utils import save_image
import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim import Adam
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from datetime import datetime
import os
import argparse  # Import argparse for CLI handling

from vne.features import image_to_features
from vne.metrics import similarity
from vne.vae import AffinityVAE, AffinityCosineLoss
from vne.encoders import Encoder3D
from vne.decoders import GaussianSplatDecoder
from vne.special.copick import CopickDataset
from vne.special.pdb import DensitySimulator

import json

# Constants
POSE_DIMS = 4
NUM_SPLATS = 768
IMAGES_PER_EPOCH = 10_000
LEARNING_RATE = 1e-5
EPOCHS = 2500
LATENT_DIMS = 8
# BETA = 1.0
# BETA = 0
BETA = 0.25
GAMMA = 0.5
# GAMMA = 5.0
# GAMMA = 5.0
BOX_SIZE = 48
SPATIAL_DIMS = 3
BATCH_SIZE = 2
VOLUME_SIZE = (BOX_SIZE,) * SPATIAL_DIMS
KLD_WEIGHT = LATENT_DIMS / np.prod(VOLUME_SIZE)
SPLAT_SIGMA_RANGE = (0.0001, 0.05)

# Argument parser to allow CLI device selection
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="Specify the GPU device (e.g., cuda:0, cuda:1)")
args = parser.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Save run parameters to JSON
def save_params_to_json(file_path):
    params = {
        "POSE_DIMS": POSE_DIMS,
        "NUM_SPLATS": NUM_SPLATS,
        "IMAGES_PER_EPOCH": IMAGES_PER_EPOCH,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "LATENT_DIMS": LATENT_DIMS,
        "BETA": BETA,
        "GAMMA": GAMMA,
        "BOX_SIZE": BOX_SIZE,
        "SPATIAL_DIMS": SPATIAL_DIMS,
        "BATCH_SIZE": BATCH_SIZE,
        "VOLUME_SIZE": VOLUME_SIZE,
        "KLD_WEIGHT": KLD_WEIGHT,
        "SPLAT_SIGMA_RANGE": SPLAT_SIGMA_RANGE,
        "DEVICE": args.device,
    }
    with open(file_path, 'w') as f:
        json.dump(params, f, indent=4)

experiment_id = int(time.time())

mlflow_logger = MLFlowLogger(
    experiment_name=f"vae_experiment_{experiment_id}",  # Name of your experiment
    tracking_uri="http://127.0.0.1:5000"  # MLflow tracking server URI
)

checkpoint_dir = f"/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/train_mlc_{experiment_id}"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save parameters to JSON in checkpoint directory
save_params_to_json(os.path.join(checkpoint_dir, "run_params.json"))

# Get PDB IDs from CopickDataset
copick_dataset = CopickDataset("/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/ml_challenge.json", boxsize=(BOX_SIZE, BOX_SIZE, BOX_SIZE), augment=False, device=DEVICE, cache_dir="/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/cache")
pdb_ids = [obj.pdb_id.lower() for obj in copick_dataset.root.pickable_objects if obj.is_particle]



cif_dir = "/mnt/czi-sci-ai/imaging-models/kyle/Data/simulated_mlc/cif_files"

# Create directory if it doesn't exist
os.makedirs(cif_dir, exist_ok=True)

def download_cif_file(pdb_id, out_dir):
    """Download a CIF file and save it locally."""
    url = f'https://files.rcsb.org/download/{pdb_id}.cif'
    cif_file_path = os.path.join(out_dir, f'{pdb_id}.cif')
    response = requests.get(url)
    if response.status_code == 200:
        with open(cif_file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {pdb_id}.cif")
    else:
        print(f"Failed to download {pdb_id}.cif")
    return cif_file_path

# Loop through PDB IDs and download CIF files
for pdb_id in pdb_ids:
    download_cif_file(pdb_id, cif_dir)

# Set up the density simulator
simulator = DensitySimulator(
    [Path(cif_dir) / Path(f"{pdb_id}.cif") for pdb_id in pdb_ids],
    box_size=BOX_SIZE,
    pixel_size=10,
)

# Filter out PDB IDs that were not successfully loaded
valid_pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id in simulator.structures]
print(f"Successfully loaded {len(valid_pdb_ids)} out of {len(pdb_ids)} PDB structures")

N_MOLECULES = len(simulator.keys())
MOLECULE_ID = [n for n in simulator.keys()]

SCALE = 1. / simulator.pixel_size

def _features(x_i, *, use_center: bool = True):
    f_i = image_to_features(x_i, scale=SCALE, use_center=use_center)
    return f_i

def _affinity(f):
    f_i, f_j = f
    return similarity(f_i, f_j)

from pathos.multiprocessing import ProcessPool
from itertools import combinations_with_replacement
import numpy.ma as ma

def similarity_matrix(*, normalize: bool = True, fill_diagonal: bool = True, use_center: bool = True):
    molecules = list(simulator.keys())
    n_molecules = len(molecules)
    n_iter = (n_molecules * (n_molecules + 1)) // 2

    affinity = np.eye(n_molecules)
    examples = []
    features = []

    with tqdm(total=n_molecules) as pbar:
        pbar.set_description("Calculating examples")
        for mol in molecules:
            x_i = simulator(mol, project=False) > 0
            examples.append(x_i)
            pbar.update(1)
            pbar.refresh()

    # Create a ProcessPool instance
    with ProcessPool(nodes=8) as pool:
        features = pool.map(_features, examples)
    
    pairs = list(combinations_with_replacement(features, 2))
    
    # Use ProcessPool for affinity calculation with 16 workers
    with ProcessPool(nodes=16) as pool:
        affinities = pool.map(_affinity, pairs)

    # Set the upper triangular matrix
    affinity[np.triu_indices(n_molecules)] = affinities

    # Make it symmetric
    affinity = affinity + np.triu(affinity).T

    # Mask the diagonal
    masked_affinity = ma.array(affinity, mask=np.eye(n_molecules))

    # Scale it
    if normalize:
        affinity = (2 * (affinity - np.min(masked_affinity)) / np.ptp(masked_affinity)) - 1.

    # Make the diagonal equal to one
    if fill_diagonal:
        np.fill_diagonal(affinity, 1.0)
   
    return affinity, np.stack(examples, axis=0)


lookup, imgs = similarity_matrix(use_center=True, fill_diagonal=True, normalize=True)

class DensitySimulatorDataset(Dataset):
    def __init__(self, simulator, pdb_ids, num_samples=IMAGES_PER_EPOCH):
        self.simulator = simulator
        self.pdb_ids = pdb_ids
        self.num_samples = num_samples
        self.pdb_to_idx = {pdb: idx for idx, pdb in enumerate(self.pdb_ids)}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        pdb_id = np.random.choice(self.pdb_ids)
        euler_angles = R.random().as_euler('xyz', degrees=True)

        try:
            image = self.simulator(pdb_id, transform_euler_angles=euler_angles, project=False)
            image = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)  # Ensure image is on the correct device
            return image, torch.tensor(self.pdb_to_idx[pdb_id], device=DEVICE)  # Ensure mol_id is on the correct device
        except Exception as e:
            print(f"Error processing PDB ID {pdb_id}: {str(e)}")
            return torch.zeros((1, BOX_SIZE, BOX_SIZE), device=DEVICE), torch.tensor(-1, device=DEVICE)  # Return tensors on the correct device



class VAEModel(LightningModule):
    def __init__(self, encoder, decoder, latent_dims, valid_pdb_ids, simulator, checkpoint_dir, logger):
        super().__init__()
        self.model = AffinityVAE(
            encoder=encoder,
            decoder=decoder,
            latent_dims=latent_dims,
            pose_channels=POSE_DIMS,
        ).to(DEVICE)
        self.simulator = simulator
        self.valid_pdb_ids = valid_pdb_ids
        self.reconstruction_loss = nn.MSELoss(reduction="mean")
        self.similarity_loss = AffinityCosineLoss(lookup=lookup, device=DEVICE)
        self.latent_dims = latent_dims
        self.beta = BETA
        self.gamma = GAMMA
        self.kld_weight = KLD_WEIGHT
        self.checkpoint_dir = checkpoint_dir
        self.mlflow_logger = logger  # MLFlowLogger

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mol_id = batch
        img = img.to(DEVICE)
        mol_id = mol_id.to(DEVICE)

        if torch.any(mol_id == -1):
            return None

        output, z, pose, mu, log_var = self(img)

        r_loss = self.reconstruction_loss(output, img)
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld = self.beta * self.kld_weight * kld
        s_loss = self.gamma * self.similarity_loss(mol_id, mu)
        loss = r_loss + s_loss + kld

        # Log each term separately
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("reconstruction_loss", r_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("kld_loss", kld, on_step=True, on_epoch=True, prog_bar=True)
        self.log("similarity_loss", s_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        return optimizer

    def validation_step(self, batch, batch_idx):
        img, mol_id = batch
        output, z, pose, mu, log_var = self(img)

        # Save intermediate splats and image montage
        with torch.no_grad():
            splats, weights, sigma = self.model.decoder.decode_splats(z, pose)
            epoch = self.current_epoch
            splat_dir = os.path.join(self.checkpoint_dir, "splats")
            montage_dir = os.path.join(self.checkpoint_dir, "montages")
            os.makedirs(splat_dir, exist_ok=True)
            os.makedirs(montage_dir, exist_ok=True)

            splat_file = f"{splat_dir}/image_{epoch}_{batch_idx}.npz"
            np.savez(
                splat_file,
                splats=splats.cpu(),
                weights=weights.cpu(),
                sigma=sigma.cpu(),
                z=z.cpu(),
                pose=pose.cpu(),
            )

            montage_file = f"{montage_dir}/image_{epoch}_{batch_idx}.png"
            pic = to_img(torch.concat([img, output], axis=0).cpu().data)
            save_image(pic, montage_file, nrow=img.shape[0])

            # Log artifacts to MLflow
            self.mlflow_logger.experiment.log_artifact(self.logger.run_id, splat_file)
            self.mlflow_logger.experiment.log_artifact(self.logger.run_id, montage_file)


def to_img(x):
    x = torch.sum(x, axis=-1)
    x = x / torch.max(torch.ravel(x))
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, BOX_SIZE, BOX_SIZE)
    return x


# Set up the model
encoder = Encoder3D(
    input_shape=VOLUME_SIZE,
    layer_channels=(8, 16, 32, 64)
).to(DEVICE)

decoder = GaussianSplatDecoder(
    VOLUME_SIZE,
    latent_dims=LATENT_DIMS,
    n_splats=NUM_SPLATS,
    output_channels=1,
    device=DEVICE,
    splat_sigma_range=SPLAT_SIGMA_RANGE,
).to(DEVICE)

# Instantiate the PyTorch Lightning model
vae_model = VAEModel(encoder, decoder, LATENT_DIMS, valid_pdb_ids, simulator, checkpoint_dir, mlflow_logger)

# Log run parameters to MLflow
mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, os.path.join(checkpoint_dir, "run_params.json"))

weights = copick_dataset.get_sample_weights()
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# dataloader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# dataloader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader = DataLoader(copick_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler)


# Set up the PyTorch Lightning trainer
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="density_sim_vae_epoch_{epoch}",
    save_top_k=-1,
    every_n_epochs=5
)

from torch.utils.data import Subset, random_split

# Define the size of the validation split as a fraction of the CopickDataset
val_split_fraction = 0.01  # 10% of CopickDataset for validation

# Calculate sizes for training and validation splits
total_copick_size = len(copick_dataset)
val_size = int(total_copick_size * val_split_fraction)
train_size = total_copick_size - val_size

# Randomly split CopickDataset into training and validation sets
copick_train_dataset, copick_val_dataset = random_split(copick_dataset, [train_size, val_size])

# Define a smaller validation DataLoader
train_dataloader = DataLoader(copick_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dataloader = DataLoader(copick_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Set up the PyTorch Lightning trainer with both train and validation DataLoaders
trainer = Trainer(
    max_epochs=EPOCHS,
    callbacks=[checkpoint_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=[int(args.device.split(':')[1])] if torch.cuda.is_available() else None,
    logger=mlflow_logger,
    default_root_dir=checkpoint_dir
)

# Train with both the training and validation DataLoaders
trainer.fit(vae_model, train_dataloader, val_dataloader)

best_checkpoint = checkpoint_callback.best_model_path
if best_checkpoint:
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, best_checkpoint)
