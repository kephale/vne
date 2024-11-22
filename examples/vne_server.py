from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import zarr
from vne.vae import AffinityVAE
from vne.encoders import Encoder3D
from vne.decoders import GaussianSplatDecoder
import copick
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logger.info("Using CUDA device")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')  # For Apple Silicon
    logger.info("Using MPS device (Apple Silicon)")
else:
    DEVICE = torch.device('cpu')
    logger.info("Using CPU device")
    

# Configurable crop size
CROP_SIZE = (48, 48, 48)  # Default crop size (can be changed dynamically)

# Define model parameters
encoder = Encoder3D(input_shape=CROP_SIZE, layer_channels=(8, 16, 32, 64)).to(DEVICE)
decoder = GaussianSplatDecoder(CROP_SIZE, latent_dims=8, n_splats=768, output_channels=1, splat_sigma_range=(0.02, 0.1), device=DEVICE)
model = AffinityVAE(encoder=encoder, decoder=decoder, latent_dims=8, pose_channels=4).to(DEVICE)

# Load the model state dict
# state_dict_path = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/train_mlc_1731606498/density_sim_vae_epoch_epoch=4.ckpt"
# state_dict_path = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/train_mlc_1732135115/density_sim_vae_epoch_epoch=39.ckpt"
# 48s
state_dict_path = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/train_mlc_1732223081/density_sim_vae_epoch_epoch=404.ckpt"
state_dict = torch.load(state_dict_path, map_location=DEVICE)
model_state_dict = state_dict["state_dict"]
updated_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
fixed_state_dict = {}
for key, value in updated_state_dict.items():
    if key.startswith("encoder."):
        fixed_state_dict[key.replace("encoder.", "encoder.model.", 1)] = value
    elif key.startswith("decoder."):
        fixed_state_dict[key] = value
    else:
        fixed_state_dict[key] = value
model.load_state_dict(fixed_state_dict)
model.eval()

# Define input data schema
class LatentVectorRequest(BaseModel):
    run: int
    voxel_spacing: int
    tomogram: str
    coordinates: tuple

@app.post("/latent-vector")
async def get_latent_vector(request: LatentVectorRequest):
    """
    Endpoint to fetch a latent vector for the given run, voxel spacing, tomogram, and coordinates.
    """
    try:
        # Load Copick configuration and Zarr data
        COPICK_CONFIG_PATH = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/ml_challenge.json"
        root = copick.from_file(COPICK_CONFIG_PATH)
        
        # Ensure the requested run exists
        if request.run >= len(root.runs):
            raise HTTPException(status_code=400, detail="Invalid run index.")
        
        # Ensure the requested voxel spacing exists
        run_data = root.runs[request.run]
        if request.voxel_spacing >= len(run_data.voxel_spacings):
            raise HTTPException(status_code=400, detail="Invalid voxel spacing index.")
        
        # Load the requested tomogram
        tomogram_data = run_data.voxel_spacings[request.voxel_spacing].get_tomogram(request.tomogram)
        z = zarr.open(tomogram_data.zarr(), "r")["0"]
        
        # Validate and adjust coordinates to make them the crop center
        coords = np.array(request.coordinates)
        half_crop_size = np.array(CROP_SIZE) // 2

        # Calculate crop bounds based on the center
        start = coords - half_crop_size
        end = coords + half_crop_size

        # Adjust bounds to stay within Zarr shape limits
        if np.any(start < 0) or np.any(end > z.shape):
            raise HTTPException(
                status_code=400,
                detail=f"Crop exceeds bounds. Adjusted start: {start}, end: {end}, zarr shape: {z.shape}"
            )
        
        # Fetch and normalize crop
        crop = z[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2],
        ]
        crop = (crop - np.mean(crop)) / (np.std(crop) + 1e-6)
        # crop = crop / crop.max()
        
        # Prepare input tensor
        input_tensor = torch.tensor(crop, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        
        # Compute latent vector
        with torch.no_grad():
            # Get latent variables and pose
            mu, log_var, pose = model.encode(input_tensor)  # Encode input to latent space
            z = model.reparameterise(mu, log_var)  # Reparameterize to sample z
            reconstructed = model.decoder(z, pose)  # Decode z and pose to reconstruct
            
        # Return latent vector and pose
        return {
            "latent_vector": z.squeeze().cpu().numpy().tolist(),
            "pose": pose.squeeze().cpu().numpy().tolist(),
        }
    
    except Exception as e:
        # Log the error
        logger.exception("Error processing request: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reconstruction")
async def get_reconstruction(request: LatentVectorRequest):
    """
    Endpoint to fetch the full reconstruction for the given run, voxel spacing, tomogram, and coordinates.
    """
    try:
        # Load Copick configuration and Zarr data
        COPICK_CONFIG_PATH = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/ml_challenge.json"
        root = copick.from_file(COPICK_CONFIG_PATH)
        
        # Ensure the requested run exists
        if request.run >= len(root.runs):
            raise HTTPException(status_code=400, detail="Invalid run index.")
        
        # Ensure the requested voxel spacing exists
        run_data = root.runs[request.run]
        if request.voxel_spacing >= len(run_data.voxel_spacings):
            raise HTTPException(status_code=400, detail="Invalid voxel spacing index.")
        
        # Load the requested tomogram
        tomogram_data = run_data.voxel_spacings[request.voxel_spacing].get_tomogram(request.tomogram)
        z = zarr.open(tomogram_data.zarr(), "r")["0"]
        
        # Validate and adjust coordinates to make them the crop center
        coords = np.array(request.coordinates)
        half_crop_size = np.array(CROP_SIZE) // 2

        # Calculate crop bounds based on the center
        start = coords - half_crop_size
        end = coords + half_crop_size

        # Adjust bounds to stay within Zarr shape limits
        if np.any(start < 0) or np.any(end > z.shape):
            raise HTTPException(
                status_code=400,
                detail=f"Crop exceeds bounds. Adjusted start: {start}, end: {end}, zarr shape: {z.shape}"
            )
        
        # Fetch and normalize crop
        crop = z[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2],
        ]
        crop = (crop - np.mean(crop)) / (np.std(crop) + 1e-6)
        # crop = crop / crop.max()
        
        # Prepare input tensor
        input_tensor = torch.tensor(crop, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        
        # Compute reconstruction
        with torch.no_grad():
            # Get latent variables and pose
            mu, log_var, pose = model.encode(input_tensor)  # Encode input to latent space
            z = model.reparameterise(mu, log_var)  # Reparameterize to sample z
            # reconstructed = model.decoder(z, pose)  # Decode z and pose to reconstruct
            reconstructed = model.decoder(z, pose, use_final_convolution=False)  # Decode z and pose to reconstruct
            # reconstructed = model.decoder(z, pose, use_final_convolution=True)  # Decode z and pose to reconstruct
            
        # Return the reconstruction as a list of numbers
        return {
            "reconstruction": reconstructed.squeeze().cpu().numpy().tolist(),
            "pose": pose.squeeze().cpu().numpy().tolist(),
        }
    
    except Exception as e:
        # Log the error
        logger.exception("Error processing reconstruction request: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

