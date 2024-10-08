import os
import string
import requests
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops
from skimage.transform import rotate

from . import shapes

CHARS = string.ascii_lowercase + string.digits
DEFAULT_FONT = None  # ImageFont.load_default()


def _download_default_font():
    """Download a default font and return the filename."""
    font_url = "https://github.com/edx/edx-fonts/raw/refs/heads/master/open-sans/fonts/Regular/OpenSans-Regular.ttf"
    font_path = os.path.join(os.path.dirname(__file__), "special", "font", "OpenSans-Regular.ttf")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(font_path), exist_ok=True)
    print(font_path)

    # Download the font if it does not exist
    if not os.path.exists(font_path):
        response = requests.get(font_url)
        response.raise_for_status()  # Raise an error if the download failed
        with open(font_path, "wb") as f:
            f.write(response.content)
        print(f"Font downloaded to: {font_path}")

    # Verify the file format by trying to load it
    try:
        font = ImageFont.truetype(font_path)
        print(f"Font successfully loaded: {font_path}")
    except OSError:
        raise OSError(f"Failed to load font from {font_path}. The file format might be invalid.")
    
    return font_path


def set_default_font(filename: Optional[os.PathLike] = None, size: int = 36):
    """Set the default font from a file."""
    print(filename)
    if not filename:
        filename = _download_default_font()

    font = ImageFont.truetype(filename, size)
    global DEFAULT_FONT
    DEFAULT_FONT = font


def check_number(input: float, max: float):
    """Enforces coordinates to an interval of valid values"""
    max = max - 64
    if input < 0:
        output = 0
    elif input > max > 0:
        output = max
    else:
        output = input
    return output


def create_example(text: str, angle: float = 0.0) -> np.ndarray:
    """Create an example character with a random translational offset and
    rotation."""
    raw = np.zeros((64, 64), dtype=np.uint8)
    img = Image.fromarray(raw)
    d = ImageDraw.Draw(img)

    offset = tuple(
        [
            (raw.shape[dim] - DEFAULT_FONT.getbbox(text)[dim]) // 2
            for dim in range(raw.ndim)
        ]
    )

    d.text(offset, text, font=DEFAULT_FONT, fill=(255))
    img_arr = np.asarray(img)
    img_rot = rotate(img_arr, angle, preserve_range=True)
    return img_rot


def create_heterogeneous_image(
    shape: Tuple[int, ...],
    n_objects: int = 100,
    return_masks: bool = False,
    rng=np.random.default_rng(),
) -> Tuple:
    """Create a big training image.

    Parameters
    ----------
    shape : tuple(int)
        The shape of the output image.
    n_objects : int
        The number of simulated objects.
    return_masks : bool
        Option to return binary masks for each object as (N, W, H) array.


    Returns
    -------
    image : np.ndarray
        The simulated image.
    bounding_boxes : list
        A list of bounding boxes for each of the objects.
    labels : np.ndarray
        A list of labels for each of the objects.
    """

    assert len(shape) == 2

    big_image = np.zeros(shape, dtype=np.uint8)
    locations = rng.integers(0, shape[0] - 64, (n_objects, 2))
    labels = rng.choice(list(CHARS), size=n_objects)
    bounding_boxes = []

    if return_masks:
        masks = np.zeros((n_objects,) + shape, dtype=np.uint8)

    for i in range(n_objects):
        char = labels[i]
        angle = rng.integers(0, 360)
        sx = slice(locations[i, 0], locations[i, 0] + 64)
        sy = slice(locations[i, 1], locations[i, 1] + 64)

        example = create_example(char, angle)

        # hack to get the rotated bounding box
        props = regionprops(example.astype(bool).astype(int))
        bbox = np.array(props[0].bbox) + np.concatenate([locations[i, :]] * 2)
        bounding_boxes.append(bbox)

        big_image[sx, sy] = np.maximum(
            big_image[sx, sy],
            example,
        )

        if return_masks:
            masks[i, sx, sy] = example.astype(bool).astype(np.uint8)

    if return_masks:
        return big_image, bounding_boxes, labels, masks

    return big_image, bounding_boxes, labels


def create_heterogeneous_image_with_shapes(
    shape: Tuple[int],
    n_objects: int = 100,
    return_masks: bool = False,
    input_image: Optional[np.ndarray] = None,
    input_bbox: Optional[np.ndarray] = None,
    input_labels: Optional[np.ndarray] = None,
    rng=np.random.default_rng(),
) -> Tuple:
    """Create a shape made of letters in a big training image.
    Parameters
    ----------
    shape : tuple(int)
        The shape of the output image.
    n_objects : int
        The number of points between which the shapes are created.
    return_masks : bool
        Option to return binary masks for each object as (N, W, H) array.
    input_image : ndarray
        (Optional) The input image on which to add the shape.
    input_bbox : ndarray
        (Optional) The input array with all the bounding boxes.
    input_labels : ndarray
        (Optional) The input array of the labels for each bounding box.

    Returns
    -------
    image : np.ndarray
        The simulated image.
    bounding_boxes : list
        A list of bounding boxes for each of the objects.
    labels : list
        A list of labels for each of the objects.
    """

    assert len(shape) == 2

    if input_image is None:
        big_image, pre_bbox, pre_labels = create_heterogeneous_image(shape)
        if pre_bbox is None or pre_labels is None:
            raise ValueError("Labels or bounding boxes missing.")
    else:
        big_image = input_image
        pre_bbox = input_bbox
        pre_labels = input_labels
    locations = rng.integers(0, shape[0] - 64, (n_objects, 2))
    labels = rng.choice(list(CHARS), size=n_objects)
    bounding_boxes = []
    x, y, _ = shapes.get_bezier_curve(locations)

    x = np.rint(x).astype(int)
    y = np.rint(y).astype(int)
    if return_masks:
        masks = np.zeros((n_objects,) + shape, dtype=np.uint8)

    # TO DO (bcg): make the character choice an option, or random
    char = labels[0]
    angle = rng.integers(0, 360)

    example = create_example(str(char), angle)

    props = regionprops(example.astype(bool).astype(int))
    for j in range(x.shape[0]):
        x[j] = check_number(x[j], shape[0])
        y[j] = check_number(y[j], shape[1])
        sx = slice(x[j], x[j] + 64)
        sy = slice(y[j], y[j] + 64)

        bbox = np.array(props[0].bbox) + np.concatenate(
            [np.array([int(x[j]), int(y[j])])] * 2
        )
        bounding_boxes.append(bbox)

        big_image[sx, sy] = np.maximum(
            big_image[sx, sy],
            example,
        )

        if return_masks:
            masks[j, sx, sy] = example.astype(bool).astype(np.uint8)

    for box in pre_bbox:
        bounding_boxes.append(box)

    np.append(labels, pre_labels)
    if return_masks:
        # TO DO (bcg): add the masks of the newly created objects
        return big_image, bounding_boxes, labels, masks

    return big_image, bounding_boxes, labels


if __name__ == "__main__":
    pass
