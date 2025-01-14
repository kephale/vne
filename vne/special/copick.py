import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import copick
from pathlib import Path
import os
import pickle
from scipy.ndimage import gaussian_filter
import random
from collections import Counter
from torch.utils.data import DistributedSampler

class CopickDataset(Dataset):
    def __init__(
        self,
        config_path: str,
        boxsize: Tuple[int, int, int] = (32, 32, 32),
        augment: bool = False,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        seed: Optional[int] = 1717,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        active_samples: int = 25000,
        refresh_epochs: int = 10
    ):
        self.config_path = config_path
        self.boxsize = boxsize
        self.augment = augment
        self.cache_dir = cache_dir
        self.device = device
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.active_samples = active_samples
        self.refresh_epochs = refresh_epochs
        self.current_epoch = 0
        
        self._set_random_seed()
        self._initialize_data_structures()
        self._discover_all_points()
        self._load_initial_samples()
        
    def _initialize_data_structures(self):
        """Initialize empty data structures."""
        self._subvolumes = []
        self._molecule_ids = []
        self._keys = []
        self._all_points: Dict[str, List[Tuple[float, float, float]]] = {}
        self._current_indices = None
        
    def _set_random_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
    def _discover_all_points(self):
        """Discover and store all available points without loading volumes."""
        print("\n=== Starting point discovery process ===")
        root = copick.from_file(self.config_path)
        voxel_spacing = 10
        
        for run_idx, run in enumerate(root.runs):
            try:
                for pick_idx, picks in enumerate(run.picks):
                    if not picks.from_tool:
                        continue
                        
                    object_name = picks.pickable_object_name
                    if object_name not in self._keys:
                        self._keys.append(object_name)
                        
                    points, _ = picks.numpy()
                    points = points / voxel_spacing
                    
                    if object_name not in self._all_points:
                        self._all_points[object_name] = []
                    self._all_points[object_name].extend(points)
                    
            except Exception as e:
                print(f"Error processing run {run_idx}: {str(e)}")
                continue
                
        print(f"\nDiscovered points for {len(self._keys)} unique objects")
        for key in self._keys:
            print(f"{key}: {len(self._all_points[key])} points")
            
    def _load_from_cache(self) -> bool:
        """Try to load data from cache if available."""
        if not self.cache_dir:
            return False
            
        cache_file = os.path.join(
            self.cache_dir,
            f"copick_cache_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}.pkl"
        )
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self._all_points = cached_data['all_points']
                self._keys = cached_data['keys']
            return True
        return False
        
    def _save_to_cache(self):
        """Save discovered points to cache."""
        if not self.cache_dir:
            return
            
        cache_file = os.path.join(
            self.cache_dir,
            f"copick_cache_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}.pkl"
        )
        
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'all_points': self._all_points,
                'keys': self._keys
            }, f)
        print(f"Saved point data to cache: {cache_file}")
        
    def _load_initial_samples(self):
        """Load initial batch of samples."""
        if self._load_from_cache():
            self._refresh_active_samples()
        else:
            self._refresh_active_samples()
            self._save_to_cache()
            
    def _refresh_active_samples(self):
        """Refresh the currently active samples."""
        print("\n=== Refreshing active samples ===")
        
        # Clear current data
        self._subvolumes = []
        self._molecule_ids = []
        
        # Calculate samples per class
        num_classes = len(self._keys)
        samples_per_class = self.active_samples // num_classes
        remaining_samples = self.active_samples % num_classes
        
        root = copick.from_file(self.config_path)
        voxel_spacing = 10
        tomogram = root.runs[0].get_voxel_spacing(voxel_spacing).tomograms[0]
        tomogram_array = tomogram.numpy()
        
        for class_idx, object_name in enumerate(self._keys):
            points = self._all_points[object_name]
            
            # Calculate number of samples for this class
            class_samples = samples_per_class + (1 if class_idx < remaining_samples else 0)
            
            # Randomly select points if we have more than needed
            if len(points) > class_samples:
                selected_points = random.sample(points, class_samples)
            else:
                selected_points = points
                
            # Load volumes for selected points
            for point in selected_points:
                try:
                    subvolume = self._extract_subvolume(tomogram_array, *point)
                    self._subvolumes.append(subvolume)
                    self._molecule_ids.append(self._keys.index(object_name))
                except ValueError as e:
                    print(f"Failed to extract subvolume for point {point}: {str(e)}")
                    
        self._subvolumes = np.array(self._subvolumes)
        self._molecule_ids = np.array(self._molecule_ids)
        
        print(f"Loaded {len(self._subvolumes)} active samples")
        
    def _extract_subvolume(self, tomogram_array, x, y, z):
        """Extract a subvolume from the tomogram."""
        half_box = np.array(self.boxsize) // 2
        x_slice = slice(int(x - half_box[0]), int(x + half_box[0]))
        y_slice = slice(int(y - half_box[1]), int(y + half_box[1]))
        z_slice = slice(int(z - half_box[2]), int(z + half_box[2]))
        
        try:
            subvolume = tomogram_array[z_slice, y_slice, x_slice]
        except IndexError as e:
            raise ValueError(f"Error extracting subvolume: {str(e)}")
        
        return self._pad_or_crop(subvolume)
        
    def _pad_or_crop(self, subvolume):
        """Ensure subvolume matches target size through padding or cropping."""
        current_shape = np.array(subvolume.shape)
        target_shape = np.array(self.boxsize)
        
        if np.all(current_shape == target_shape):
            return subvolume
        
        result = np.zeros(target_shape, dtype=subvolume.dtype)
        
        for dim in range(3):
            if current_shape[dim] < target_shape[dim]:
                pad_width = (target_shape[dim] - current_shape[dim]) // 2
                start = pad_width
                end = start + current_shape[dim]
            else:
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
        
    def update_epoch(self, epoch: int):
        """Update the current epoch and refresh samples if needed."""
        self.current_epoch = epoch
        if epoch % self.refresh_epochs == 0:
            self._refresh_active_samples()
            
    def __len__(self):
        return len(self._subvolumes)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subvolume = self._subvolumes[idx]
        molecule_idx = self._molecule_ids[idx]
        
        if self.augment:
            subvolume = self._augment_subvolume(subvolume)
            
        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        return subvolume, torch.tensor(molecule_idx)
        
    def _augment_subvolume(self, subvolume):
        """Apply data augmentation to a subvolume."""
        if random.random() < 0.5:
            subvolume = self._brightness(subvolume)
        if random.random() < 0.5:
            subvolume = self._gaussian_blur(subvolume)
        if random.random() < 0.5:
            subvolume = self._intensity_scaling(subvolume)
        if random.random() < 0.5:
            subvolume = self._contrast_adjustment(subvolume)
        subvolume, _ = self._rotation_180_degrees(subvolume, subvolume)
        return subvolume

    def _brightness(self, volume, max_delta=0.5):
        delta = np.random.uniform(-max_delta, max_delta)
        return volume + delta

    def _gaussian_blur(self, volume, sigma_range=(0.75, 1.25)):
        sigma = np.random.uniform(*sigma_range)
        return gaussian_filter(volume, sigma=sigma)

    def _intensity_scaling(self, volume, intensity_range=(0.5, 1.5)):
        intensity_factor = np.random.uniform(*intensity_range)
        return volume * intensity_factor

    def _contrast_adjustment(self, volume, contrast_range=(0.5, 1.5)):
        contrast_factor = np.random.uniform(*contrast_range)
        mean = np.mean(volume)
        return mean + contrast_factor * (volume - mean)

    def _rotation_180_degrees(self, volume, target, augment_probability=0.8):
        if np.random.rand() < augment_probability:
            chosen_axis = (0, 2)  # Rotate around x-z plane
            volume = np.rot90(volume, k=2, axes=chosen_axis)
            target = np.rot90(target, k=2, axes=chosen_axis)
        return volume, target

    def keys(self) -> List[str]:
        return self._keys

    def get_pickable_objects(self):
        root = copick.from_file(self.config_path)
        return root.pickable_objects

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

        return torch.stack(examples, axis=0).to(self.device), [self._keys[idx] for idx in examples_class]