import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import copick
from pathlib import Path

class CopickDataset(Dataset):
    def __init__(
        self,
        config_path: str,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        augment: bool = False
    ):
        self.root = copick.from_file(config_path)
        self.boxsize = boxsize
        self.augment = augment
        
        self._subvolumes = []
        self._molecule_ids = []
        self._keys = []
        
        self._load_data()

        if len(self._subvolumes) == 0:
            raise ValueError("No valid subvolumes found in the dataset. Please check your Copick configuration and ensure there are valid picks and tomograms.")

    def _load_data(self):
        voxel_spacing = 10  # TODO: Consider making this a parameter
        for run in self.root.runs:
            print(f"Processing Run: {run}")
            try:
                tomogram = run.get_voxel_spacing(voxel_spacing).tomograms[0]
                # Load the entire tomogram array once
                tomogram_array = tomogram.numpy()
                print(f"Loaded tomogram with shape: {tomogram_array.shape}")
            except AttributeError:
                print(f"Warning: Could not find tomogram for run {run.name}. Skipping this run.")
                continue
            
            for picks in run.picks:
                if picks.from_tool:  # Only use tool-generated picks
                    object_name = picks.pickable_object_name
                    points, _ = picks.numpy()
                    print(f"Processing {len(points)} points for {object_name}")

                    # Adjust for voxel spacing
                    points = points / voxel_spacing
                    
                    for point in points:
                        try:
                            x, y, z = point
                            subvolume = self._extract_subvolume(tomogram_array, x, y, z)
                            self._subvolumes.append(subvolume)
                            
                            if object_name not in self._keys:
                                self._keys.append(object_name)
                            
                            self._molecule_ids.append(self._keys.index(object_name))
                        except ValueError as e:
                            print(f"Warning: Could not extract subvolume for point {point}. Error: {str(e)}")
        
        self._subvolumes = np.array(self._subvolumes)
        self._molecule_ids = np.array(self._molecule_ids)
        
        print(f"Loaded {len(self._subvolumes)} subvolumes with {len(self._keys)} unique object types.")

    def _extract_subvolume(self, tomogram_array, x, y, z):
        half_box = np.array(self.boxsize) // 2
        x_slice = slice(int(x - half_box[0]), int(x + half_box[0]))
        y_slice = slice(int(y - half_box[1]), int(y + half_box[1]))
        z_slice = slice(int(z - half_box[2]), int(z + half_box[2]))
        
        try:
            subvolume = tomogram_array[z_slice, y_slice, x_slice]
        except IndexError as e:
            raise ValueError(f"Error extracting subvolume: {str(e)}. Check if the point ({x}, {y}, {z}) is within the tomogram bounds.")
        
        # Pad or crop to ensure consistent size
        subvolume = self._pad_or_crop(subvolume)
        
        return subvolume

    def _pad_or_crop(self, subvolume):
        current_shape = np.array(subvolume.shape)
        target_shape = np.array(self.boxsize)
        
        if np.all(current_shape == target_shape):
            return subvolume
        
        result = np.zeros(target_shape, dtype=subvolume.dtype)
        
        for dim in range(3):
            if current_shape[dim] < target_shape[dim]:
                # Pad
                pad_width = (target_shape[dim] - current_shape[dim]) // 2
                start = pad_width
                end = start + current_shape[dim]
            else:
                # Crop
                crop = (current_shape[dim] - target_shape[dim]) // 2
                start = crop
                end = start + target_shape[dim]
            
            if dim == 0:
                result[:] = subvolume[start:end]
            elif dim == 1:
                result[:, :] = subvolume[:, start:end]
            else:
                result[:, :, :] = subvolume[:, :, start:end]
        
        return result

    def __len__(self):
        return len(self._subvolumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subvolume = self._subvolumes[idx]
        molecule_idx = self._molecule_ids[idx]

        if self.augment:
            subvolume = self._augment_subvolume(subvolume)

        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)  # Add small epsilon to avoid division by zero
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        return subvolume, molecule_idx

    def _augment_subvolume(self, subvolume):
        # Implement augmentation logic here (e.g., random rotations, flips)
        # For this example, we'll just add some random noise
        noise = np.random.normal(0, 0.1, subvolume.shape)
        return subvolume + noise

    def keys(self) -> List[str]:
        return self._keys

    def examples(self) -> Tuple[torch.Tensor, List[str]]:
        x_idx = set()
        x_complete = set(range(len(self._keys)))
        examples = []
        examples_class = []
        idx = 0

        while x_complete.difference(x_idx) != set():
            vol, mol_idx = self[idx]
            if mol_idx not in x_idx:
                x_idx.add(mol_idx)
                examples.append(vol)
                examples_class.append(mol_idx)
            idx += 1

        return torch.stack(examples, axis=0), [self._keys[idx] for idx in examples_class]