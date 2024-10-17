import torch
import zarr
import numpy as np
from tqdm import tqdm
from vne.vae import AffinityVAE
from vne.encoders import Encoder3D
from vne.decoders import GaussianSplatDecoder
import copick
from numcodecs import Blosc

def load_model(model_path, latent_dims, volume_size, num_splats=768, output_channels=1, device="cpu"):
    encoder = Encoder3D(
        input_shape=volume_size,
        layer_channels=(8, 16, 32, 64)
    ).to(device)
    
    decoder = GaussianSplatDecoder(
        volume_size,
        latent_dims=latent_dims,
        n_splats=num_splats,
        output_channels=output_channels,
        device=device,
        splat_sigma_range=(0.01, 0.1),
    )
    
    model = AffinityVAE(
        encoder=encoder,
        decoder=decoder,
        latent_dims=latent_dims,
        pose_channels=4,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def sliding_window_3d(array, window_size, stride=1):
    shape = array.shape
    for i in range(0, shape[0] - window_size[0] + 1, stride):
        for j in range(0, shape[1] - window_size[1] + 1, stride):
            for k in range(0, shape[2] - window_size[2] + 1, stride):
                yield (
                    array[i:i+window_size[0], j:j+window_size[1], k:k+window_size[2]],
                    (i, j, k)
                )

def process_tomogram(tomogram, feature_type, model_path, window_size, latent_dims, batch_size=16):
    """
    Process a tomogram using the VAE model to compute embeddings in batches.
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, latent_dims, window_size, device=device).to(device)

    # Access the tomogram zarr array
    image = zarr.open(tomogram.zarr(), mode='r')['0']
    local_image = image[:]

    # Prepare output zarr array for embeddings
    copick_features = tomogram.get_features(feature_type)
    if copick_features is None:
        copick_features = tomogram.new_features(feature_type)
    feature_store = copick_features.zarr()

    # Create the zarr array to store features
    out_array = zarr.create(
        shape=(*image.shape, latent_dims),
        chunks=(*image.chunks, latent_dims),
        dtype='float32',
        compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
        store=feature_store,
        overwrite=True
    )

    # Process the tomogram with sliding window and batching
    stride = 16  # Adjust this if needed
    patches = []
    indices = []
    
    with torch.no_grad():
        # Note that the sliding window uses the local image
        for window, (i, j, k) in tqdm(sliding_window_3d(local_image, window_size, stride=stride),
                                      total=((image.shape[0]-window_size[0])//stride + 1) *
                                            ((image.shape[1]-window_size[1])//stride + 1) *
                                            ((image.shape[2]-window_size[2])//stride + 1)):

            # Collect patches into a batch
            patches.append(window)
            indices.append((i, j, k))

            if len(patches) == batch_size:
                # Convert batch to tensor and run through model
                input_tensor = torch.from_numpy(np.array(patches)).float().unsqueeze(1).to(device)  # Shape: (batch_size, 1, 32, 32, 32)

                # Get embeddings from VAE model
                _, _, _, mu, _ = model(input_tensor)

                # Write embeddings to the zarr array
                for idx, (i, j, k) in enumerate(indices):
                    out_array[i, j, k, :] = mu[idx].cpu().numpy().squeeze()

                # Reset for next batch
                patches = []
                indices = []

        # Process the remaining patches (if any)
        if patches:
            input_tensor = torch.from_numpy(np.array(patches)).float().unsqueeze(1).to(device)
            _, _, _, mu, _ = model(input_tensor)
            for idx, (i, j, k) in enumerate(indices):
                out_array[i, j, k, :] = mu[idx].cpu().numpy().squeeze()

    print(f"Features saved under feature type '{feature_type}'")
    return copick_features


if __name__ == "__main__":
    copick_root = copick.from_file("/root/git/quantumjot/vne/synthetic_data_10439_CDP.json")
    tomogram = copick_root.runs[0].voxel_spacings[0].get_tomogram("wbp")

    process_tomogram(
        tomogram=tomogram,
        feature_type="VAEEmbedding12",
        model_path="/root/git/quantumjot/vne/splat_autoencoder_pdb=6_beta=1.0_gamma=1.0_N=768_sz=32_epoch=60_model=MODEL.pth",
        window_size=(32, 32, 32),
        latent_dims=8
    )
