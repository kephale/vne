import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
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
        max_samples: Optional[int] = None
    ):
        self.config_path = config_path
        self.boxsize = boxsize
        self.augment = augment
        self.cache_dir = cache_dir
        self.device = 'cpu'
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.max_samples = max_samples
        self._set_random_seed()
        self._subvolumes = []
        self._molecule_ids = []
        self._keys = []
        self._load_or_process_data()
        self._compute_sample_weights()

    def _compute_sample_weights(self):
        """
        Compute sample weights based on class frequency for balancing.
        """
        class_counts = Counter(self._molecule_ids)
        total_samples = len(self._molecule_ids)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        self.sample_weights = [class_weights[mol_id] for mol_id in self._molecule_ids]

    def _set_random_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _load_or_process_data(self):
        # If cache_dir is None, process data directly without caching
        if self.cache_dir is None:
            print("Cache directory not specified. Processing data without caching...")
            self._load_data()
            return
            
        # If cache_dir is specified, use caching logic
        cache_file = os.path.join(
            self.cache_dir, 
            f"copick_cache_{self.boxsize[0]}x{self.boxsize[1]}x{self.boxsize[2]}.pkl"
        )
        
        # Only rank 0 process should create the cache
        if self.rank is None or self.rank == 0:
            if not os.path.exists(cache_file):
                print("Processing data and creating cache...")
                self._load_data()
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'subvolumes': self._subvolumes,
                        'molecule_ids': self._molecule_ids,
                        'keys': self._keys
                    }, f)
                print(f"Cached data saved to {cache_file}")
        
        # Wait for rank 0 to finish creating cache if needed
        if self.world_size is not None:
            torch.distributed.barrier()
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                
                # Apply max_samples limit if specified
                if self.max_samples is not None:
                    # Randomly select max_samples
                    total_samples = len(cached_data['subvolumes'])
                    if total_samples > self.max_samples:
                        indices = np.random.choice(
                            total_samples, 
                            self.max_samples, 
                            replace=False
                        )
                        self._subvolumes = np.array(cached_data['subvolumes'])[indices]
                        self._molecule_ids = np.array(cached_data['molecule_ids'])[indices]
                        self._keys = cached_data['keys']
                        print(f"Randomly selected {self.max_samples} samples from {total_samples} total samples")
                else:
                    self._subvolumes = cached_data['subvolumes']
                    self._molecule_ids = cached_data['molecule_ids']
                    self._keys = cached_data['keys']

        # If using distributed training, partition the dataset
        if self.rank is not None and self.world_size is not None:
            total_size = len(self._subvolumes)
            indices = list(range(total_size))
            split_size = total_size // self.world_size
            start_idx = self.rank * split_size
            end_idx = start_idx + split_size if self.rank != self.world_size - 1 else total_size
            
            self._subvolumes = self._subvolumes[start_idx:end_idx]
            self._molecule_ids = self._molecule_ids[start_idx:end_idx]


    def _load_data(self):
        print("\n=== Starting data loading process ===")
        print(f"Config path: {self.config_path}")
        
        # Load copick root with detailed error handling
        try:
            root = copick.from_file(self.config_path)
            print(f"Successfully loaded copick root")
            print(f"Number of runs found: {len(root.runs)}")
        except Exception as e:
            print(f"Failed to load copick root: {str(e)}")
            return

        voxel_spacing = 10
        
        for run_idx, run in enumerate(root.runs):
            print(f"\n--- Processing Run {run_idx}: {run.name} ---")
            
            # Try to load tomogram with detailed error handling
            try:
                tomogram = run.get_voxel_spacing(voxel_spacing).tomograms[0]
                tomogram_array = tomogram.numpy()
                print(f"Successfully loaded tomogram with shape: {tomogram_array.shape}")
            except IndexError:
                print(f"No tomograms found in run {run.name}")
                continue
            except AttributeError as e:
                print(f"Error accessing tomogram attributes: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error loading tomogram: {str(e)}")
                continue

            # Process picks with detailed logging
            print(f"Number of pick sets: {len(run.picks)}")
            for pick_idx, picks in enumerate(run.picks):
                print(f"\nProcessing pick set {pick_idx}")
                print(f"From tool: {picks.from_tool}")
                if not picks.from_tool:
                    print("Skipping non-tool picks")
                    continue
                    
                object_name = picks.pickable_object_name
                print(f"Object name: {object_name}")
                
                try:
                    points, _ = picks.numpy()
                    print(f"Found {len(points)} points")
                    
                    points = points / voxel_spacing
                    successful_extractions = 0
                    
                    for point_idx, point in enumerate(points):
                        try:
                            x, y, z = point
                            subvolume = self._extract_subvolume(tomogram_array, x, y, z)
                            self._subvolumes.append(subvolume)
                            
                            if object_name not in self._keys:
                                self._keys.append(object_name)
                            
                            self._molecule_ids.append(self._keys.index(object_name))
                            successful_extractions += 1
                            
                            if point_idx % 100 == 0:  # Log progress every 100 points
                                print(f"Processed {point_idx + 1}/{len(points)} points")
                                
                        except ValueError as e:
                            print(f"Failed to extract subvolume for point {point}: {str(e)}")
                    
                    print(f"Successfully extracted {successful_extractions}/{len(points)} subvolumes")
                    
                except Exception as e:
                    print(f"Error processing picks: {str(e)}")
                    continue
        
        self._subvolumes = np.array(self._subvolumes)
        self._molecule_ids = np.array(self._molecule_ids)

        # TODO check that random seeds are handled properly
        if self.max_samples is not None and len(self._subvolumes) > self.max_samples:
            indices = np.random.choice(len(self._subvolumes), self.max_samples, replace=False)
            self._subvolumes = np.array(self._subvolumes)[indices]
            self._molecule_ids = np.array(self._molecule_ids)[indices]
        
        print("\n=== Data loading summary ===")
        print(f"Total subvolumes loaded: {len(self._subvolumes)}")
        print(f"Unique object types: {len(self._keys)}")
        print(f"Object types: {self._keys}")
        
        # Clear the reference to root to avoid pickling issues
        del root

    def _extract_subvolume(self, tomogram_array, x, y, z):
        half_box = np.array(self.boxsize) // 2
        x_slice = slice(int(x - half_box[0]), int(x + half_box[0]))
        y_slice = slice(int(y - half_box[1]), int(y + half_box[1]))
        z_slice = slice(int(z - half_box[2]), int(z + half_box[2]))
        
        try:
            subvolume = tomogram_array[z_slice, y_slice, x_slice]
        except IndexError as e:
            raise ValueError(f"Error extracting subvolume: {str(e)}. Check if the point ({x}, {y}, {z}) is within the tomogram bounds.")
        
        return self._pad_or_crop(subvolume)

    def _pad_or_crop(self, subvolume):
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

    def __len__(self):
        return len(self._subvolumes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subvolume = self._subvolumes[idx]
        molecule_idx = self._molecule_ids[idx]

        if self.augment:
            subvolume = self._augment_subvolume(subvolume)

        subvolume = (subvolume - np.mean(subvolume)) / (np.std(subvolume) + 1e-6)
        subvolume = torch.as_tensor(subvolume[None, ...], dtype=torch.float32)
        return subvolume, torch.tensor(molecule_idx)  # Return on CPU
        
    def get_sample_weights(self):
        """
        Returns the computed sample weights for use in a WeightedRandomSampler.
        """
        return self.sample_weights

    def _augment_subvolume(self, subvolume):
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