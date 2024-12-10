from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.utils.data import DataLoader, random_split
from vne.special.copick import CopickDataset
import torch
import os

# Constants
BOX_SIZE = 48
BATCH_SIZE = 2

# Initialize the FastAPI app
app = FastAPI()

# Dataset configuration
dataset_path = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/ml_challenge.json"
cache_dir = "/mnt/czi-sci-ai/imaging-models/kyle/experiments/cryolens_mlchallenge/cache"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
copick_dataset = CopickDataset(
    dataset_path, 
    boxsize=(BOX_SIZE, BOX_SIZE, BOX_SIZE), 
    augment=False, 
    device=device, 
    cache_dir=cache_dir
)

# Create DataLoaders
val_split_fraction = 0.01
total_copick_size = len(copick_dataset)
val_size = int(total_copick_size * val_split_fraction)
train_size = total_copick_size - val_size

copick_train_dataset, copick_val_dataset = random_split(copick_dataset, [train_size, val_size])

dataloader = DataLoader(copick_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Data model for the response class
class DataResponse(BaseModel):
    image: list  # Flattened tensor for simplicity
    mol_id: int

@app.get("/data/{index}", response_model=DataResponse)
def get_data(index: int):
    """
    Fetch a specific dataset item by index.

    Args:
        index (int): The index of the item to fetch.

    Returns:
        DataResponse: The requested dataset item.
    """
    try:
        # Validate index
        if index < 0 or index >= len(copick_train_dataset):
            raise HTTPException(status_code=404, detail="Index out of range")

        # Access the dataset item
        image, mol_id = copick_train_dataset[index]

        # Convert to lists for JSON serialization
        image_list = image.cpu().numpy().flatten().tolist()
        return DataResponse(image=image_list, mol_id=mol_id.item())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8017)
