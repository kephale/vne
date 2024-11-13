import os
import gemmi
import numpy as np
from scipy.spatial.transform import Rotation as R
from .ctf import contrast_transfer_function
from typing import Tuple, Union, List
from pathlib import Path


AXES = ["Cartn_x", "Cartn_y", "Cartn_z"]


def pdb_to_coordinates(filename: os.PathLike) -> np.ndarray:
    """Read a PDB file and return the atomic coordinates.

    Parameters
    ----------
    filename : PathLike
        A filename for the PDBx/mmCIF file describing the atomic coordinates.

    Returns
    -------
    coords : np.ndarray (N, 3)
        A numpy array of the cartesian coordinates of the atoms of the model.

    Notes
    -----
    This is super basic, and does not check for mutliple chains, cofactors etc.
    """

    doc = gemmi.cif.read_file(filename)  # copy all the data from mmCIF file
    block = doc.sole_block()  # mmCIF has exactly one block

    data = block.find("_atom_site.", AXES)

    coords = np.stack(
        [[float(r) for r in data.column(idx)] for idx in range(len(AXES))],
        axis=-1,
    )

    # center the molecule in XYZ
    centroids = np.mean(coords, axis=0)
    coords = coords - centroids

    return coords


class DensitySimulator:
    """Simulate a Cryo-EM image using atomic coordinates from PDB files.

    Parameters
    ----------
    filename : PathLike
        The PDB filename to use to extract atomic coordinates.
    pixel_size : float
        The pixel size for the image in angstroms.
    box_size : float
        The size of the box in pixels for image generation.
    add_poission_noise : bool, default = True
        Add shot noise to the final image.

    Returns
    -------
    density : np.ndarray (N, N)
        The simulated projection of the electron density.
    """

    def __init__(
        self,
        filenames: Union[List[os.PathLike], os.PathLike],
        pixel_size: float = 1.0,
        box_size: int = 128,
        defocus: float = 5e3,
    ):
        if isinstance(filenames, list):
            filenames = [Path(f) for f in filenames]
        else:
            filenames = Path(filenames)

            if filenames.is_dir():
                filenames = [
                    f for f in filenames.iterdir() if f.suffix == ".cif"
                ]
            else:
                filenames = [Path(filenames)]

        self.filenames = filenames
        self.pixel_size = pixel_size
        self.box_size = box_size

        self.structures = {}
        for filename in self.filenames:
            try:
                pdb_id = filename.stem  # This gets the filename without the extension
                self.structures[pdb_id] = pdb_to_coordinates(str(filename))
                # logging.debug(f"Successfully loaded structure for {pdb_id}")
            except Exception as e:
                # logging.error(f"Failed to load structure for {filename.name}: {str(e)}")
                pass

        # logging.info(f"Loaded {len(self.structures)} structures out of {len(self.filenames)} files")

        self.ctf = contrast_transfer_function(
            defocus=defocus,
            box_size=box_size * 2,
            pixel_size=pixel_size,
        )

    def keys(self):
        return list(self.structures.keys())

    def __call__(
        self,
        key: str,
        transform_euler_angles: list = [0, 0, 0],
        transform_translate: list = [0, 0, 0],
        project: bool = True,
        add_poisson_noise: bool = False,
    ) -> Tuple[np.ndarray]:
        # get the atomic coordinates
        coords = self.structures[key]

        # do a transform
        r = R.from_euler("xyz", transform_euler_angles, degrees=True)
        coords = r.apply(coords)

        # centre the molecule, ish
        pad = self.box_size // 2
        data = coords / self.pixel_size + [pad, pad, pad]

        # discretize the atomic coords assuming 1 px == 1 Angstrom
        density, _ = np.histogramdd(
            data,
            bins=self.box_size,
            range=tuple([(0, self.box_size - 1)] * 3),
        )

        if not project:
            assert density.ndim == 3
            return density

        density = np.sum(density, axis=-1)
        density = density + 1.0
        assert density.ndim == 2

        return density
