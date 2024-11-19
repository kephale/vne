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

# Define model parameters
encoder = Encoder3D(input_shape=(96, 96, 96), layer_channels=(8, 16, 32, 64)).to(DEVICE)
decoder = GaussianSplatDecoder((96, 96, 96), latent_dims=8, n_splats=768, output_channels=1, splat_sigma_range=(0.02, 0.1), device=DEVICE)
model = AffinityVAE(encoder=encoder, decoder=decoder, latent_dims=8, pose_channels=4).to(DEVICE)

# Load the model state dict
state_dict_path = "/Users/kharrington/Data/vne/mlc/density_sim_vae_epoch_epoch=4.ckpt"
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
        COPICK_CONFIG_PATH = "/Users/kharrington/Data/copick/CZCDP_10048_local.json"
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
        
        # Validate the coordinates and crop size
        crop_size = (96, 96, 96)
        coords = request.coordinates
        if any(c < 0 or c + cs > s for c, cs, s in zip(coords, crop_size, z.shape)):
            raise HTTPException(status_code=400, detail="Invalid coordinates or crop size.")
        
        # Fetch and normalize crop
        crop = z[
            coords[0]:coords[0] + crop_size[0],
            coords[1]:coords[1] + crop_size[1],
            coords[2]:coords[2] + crop_size[2],
        ]
        crop = crop / crop.max()
        
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

