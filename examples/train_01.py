
from vne.features import image_to_features
from vne.metrics import similarity
from vne.special import pdb
# from vne.vae import ShapeVAE, ShapeSimilarityLoss

from vne.vae import AffinityVAE, AffinityCosineLoss
# from vne.loss import AffinityKLLoss
from vne.encoders import Encoder3D
from vne.decoders import GaussianSplatDecoder

from vne.special.copick import CopickDataset


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps
from pathlib import Path
from tqdm import tqdm

from skimage.util import montage

from pathos.multiprocessing import ProcessPool
from itertools import combinations_with_replacement
import numpy.ma as ma

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from torch.distributions import MultivariateNormal 
from torch.distributions.kl import kl_divergence 

plt.rcParams.update(
    {
        "font.size": 22,
    }
)


ETA = 1.0 # this is the MI loss scalar
SOAP_BOX_SIZE = 128
CTF_PADDING = 64
USE_MU_SIMILARITY_LOSS = False # use the mean rather than sample for calculating the similarity loss


POSE_DIMS = 4
NUM_SPLATS = 1024 # 64 # 1024 is good for molecules
IMAGES_PER_EPOCH = 10_000
LEARNING_RATE = 1e-3
EPOCHS = 150
LATENT_DIMS = 8
BETA = 1.0
GAMMA = 1.0
BOX_SIZE = 32
SPATIAL_DIMS = 3
BATCH_SIZE = 32 #256
VOLUME_SIZE = (BOX_SIZE,) * SPATIAL_DIMS
KLD_WEIGHT =  LATENT_DIMS / np.prod(VOLUME_SIZE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(DEVICE)

# dataset = None
# dataset = CopickDataset("path/to/config.json", boxsize=(32, 32, 32), augment=True)
# dataset = CopickDataset("/Users/kharrington/Data/copick/synthetic_data_10439_CDP.json", boxsize=(32, 32, 32), augment=True)
dataset = CopickDataset("./synthetic_data_10439_CDP.json", boxsize=(BOX_SIZE, BOX_SIZE, BOX_SIZE), augment=True, cache_dir="./dataset_cache")

# simulator

OUTPUT_CHANNELS = 1

import os
import requests

# List of PDB codes or CIF filenames to fetch
pdb_ids = [obj.pdb_id.lower() for obj in dataset.root.pickable_objects if obj.is_particle]

# Define the directory for CIF files
cif_dir = 'cif_files'

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

# List of downloaded CIF files
CIF_FILES = [f"{pdb_id}.cif" for pdb_id in pdb_ids]


from pathlib import Path

simulator = pdb.DensitySimulator(
    [Path("./cif_files") / Path(f.lower()) for f in CIF_FILES],
    box_size=SOAP_BOX_SIZE,
    pixel_size=int(256 // SOAP_BOX_SIZE),
)


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

encoder = Encoder3D(
    input_shape=VOLUME_SIZE,
    layer_channels=(8, 16, 32, 64)
).to(DEVICE)

decoder = GaussianSplatDecoder(
    VOLUME_SIZE,
    latent_dims=LATENT_DIMS,
    n_splats=NUM_SPLATS,
    output_channels=OUTPUT_CHANNELS,
    device=DEVICE,
    splat_sigma_range=(0.01, 0.1),
)

model = AffinityVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dims=LATENT_DIMS,
    pose_channels=POSE_DIMS,
).cuda()

print(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(f"Trainable params: {params}")

#if dataset.model_type == "grandmodel":
if True:
    def to_img(x):
    #     x = 0.5 * (x + 1)
        x = torch.sum(x, axis=-1) 
        x = x / torch.max(torch.ravel(x))
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, BOX_SIZE, BOX_SIZE)
        return x
else:
    def to_img(x):
    #     x = 0.5 * (x + 1)
        x = torch.mean(x, axis=-1) 
        # x = x / torch.max(torch.ravel(x))
        x = x.clamp(-1, 1)
        x = (x + 1.0) / 2.0
        x = x.view(x.size(0), 1, BOX_SIZE, BOX_SIZE)
        return x


dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
)

val_img, val_class = dataset.examples()
val_img = val_img.to(DEVICE)

val_class = pdb_ids

val_img.shape

from vne.utils.anneal import CyclicAnnealing


anneal = CyclicAnnealing(
    n_iterations=(IMAGES_PER_EPOCH * EPOCHS) // BATCH_SIZE,
)

y = [anneal.step() for _ in range (anneal.n_iterations)]

anneal.reset()

reconstruction_loss = nn.MSELoss(reduction="mean")


similarity_loss = AffinityCosineLoss(lookup=lookup, device=DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


training_iter = 0
ramp_max_iter = (EPOCHS / 2) * (IMAGES_PER_EPOCH / BATCH_SIZE)

for epoch in range(EPOCHS):
    total_loss = 0
    recon_loss = 0 
    aff_loss = 0
    kld_loss = 0

    for data in dataloader:
        img, mol_id = data
        img = Variable(img).cuda()
        mol_id = Variable(mol_id).cuda()

        # ===================forward=====================
        output, z, pose, mu, log_var = model(img)
        print(f"z shape: {z.shape}, pose shape: {pose.shape}, mu shape: {mu.shape}, log_var shape: {log_var.shape}")

        # ramp beta up
        beta = BETA * anneal.step()
        
        # reconstruction loss
        r_loss = reconstruction_loss(output, img)
        
        # kl loss
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld = beta * KLD_WEIGHT * kld

        # similarity loss
        s_loss = GAMMA * similarity_loss(mol_id, mu)
        loss = r_loss + s_loss + kld
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        recon_loss += r_loss.data
        aff_loss += s_loss.data 
        kld_loss += kld.data

        training_iter += 1.0

    # ===================log========================
    print(
        f"epoch [{epoch+1:03d}/{EPOCHS:03d}], beta: {beta:.2f}, loss: {total_loss:.4f}, "
        f"recon: {recon_loss:.4f}, "
        f"affinity: {aff_loss:.4f}, "
        f"kld: {kld_loss:.4f}, "
    )

    if epoch % 20 == 0:
        torch.save(
            model.state_dict(), 
            f"./splat_autoencoder_pdb={N_MOLECULES}_"
            f"beta={BETA}_gamma={GAMMA}_N={NUM_SPLATS}_"
            f"sz={BOX_SIZE}_epoch={epoch}_model=MODEL.pth"
            # f"model={dataset.model_type.upper()}.pth"
        )

    with torch.no_grad():
        output, z, pose, mu, log_var = model(val_img)
        splats, weights, sigma = model.decoder.decode_splats(z, pose)
        np.savez(
            f"./training/splats/image_{epoch}.npz",
            splats=splats.cpu(), 
            weights=weights.cpu(), 
            sigma=sigma.cpu(), 
            z=z.cpu(), 
            pose=pose.cpu(),
        )
        pic = to_img(torch.concat([val_img, output], axis=0).cpu().data)
        save_image(pic, f"./training/montages/image_{epoch}.png", nrow=val_img.shape[0])

