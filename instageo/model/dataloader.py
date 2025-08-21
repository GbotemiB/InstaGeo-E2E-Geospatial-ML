# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Dataloader Module."""

import os
import random
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from absl import logging
from PIL import Image
from torchvision import transforms


def open_mf_tiff_dataset(band_files: dict[str, Any]) -> xr.Dataset:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.

    Returns:
        (xr.Dataset, xr.Dataset | None, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files, (optionally) the masks, and the CRS
    """
    band_paths = list(band_files["tiles"].values())
    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
        mask_and_scale=False,  # Scaling will be applied manually
    )
    bands_dataset.band_data.attrs["scale_factor"] = 1
    return bands_dataset


def apply_multispectral_color_jitter(
    ims: List[Image.Image], 
    brightness: float = 0.1, 
    contrast: float = 0.1, 
    saturation: float = 0.1, 
    hue: float = 0.05
) -> List[Image.Image]:
    """Apply color jitter only to visible spectrum channels (RGB) of multispectral data.
    
    This function safely applies color augmentation to multispectral data by only
    modifying the first three channels (Blue, Green, Red) while preserving the
    non-visible spectrum channels (NIR, SWIR1, SWIR2) unchanged.
    
    Args:
        ims (List[Image.Image]): List of PIL Image objects representing multispectral channels.
        brightness (float): Brightness jitter factor.
        contrast (float): Contrast jitter factor.  
        saturation (float): Saturation jitter factor.
        hue (float): Hue jitter factor.
        
    Returns:
        List[Image.Image]: List of images with color jitter applied only to RGB channels.
    """
    if len(ims) < 3:
        # For fewer than 3 channels, only apply brightness and contrast
        # (saturation and hue don't make sense for grayscale)
        color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        return [color_jitter(im) for im in ims]
    
    # For multispectral data with 3+ channels
    result_ims = []
    
    # Create RGB composite from first 3 channels for color operations
    if len(ims) >= 3:
        # Convert first 3 channels to RGB mode for color jitter
        rgb_channels = []
        for i in range(3):
            # Convert to RGB mode if needed
            if ims[i].mode != 'RGB':
                # Convert single channel to RGB by duplicating across channels
                rgb_im = Image.merge('RGB', [ims[i], ims[i], ims[i]])
            else:
                rgb_im = ims[i]
            rgb_channels.append(rgb_im)
        
        # Apply color jitter to each RGB channel separately
        color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        
        for i in range(3):
            # Apply color jitter and extract the first channel back to grayscale
            jittered_rgb = color_jitter(rgb_channels[i])
            # Convert back to grayscale by taking the first channel
            if jittered_rgb.mode == 'RGB':
                jittered_gray = jittered_rgb.split()[0]
            else:
                jittered_gray = jittered_rgb
            result_ims.append(jittered_gray)
        
        # Keep remaining channels unchanged (NIR, SWIR1, SWIR2, etc.)
        for i in range(3, len(ims)):
            result_ims.append(ims[i])
    
    return result_ims


def random_crop_and_flip(
    ims: List[Image.Image], label: Image.Image, im_size: int
) -> Tuple[List[Image.Image], Image.Image]:
    """Apply random cropping and flipping transformations to the given images and label.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.

    Returns:
        Tuple[List[Image.Image], Image.Image]: A tuple containing the transformed list of
        images and label.
    """
    i, j, h, w = transforms.RandomCrop.get_params(ims[0], (im_size, im_size))

    ims = [transforms.functional.crop(im, i, j, h, w) for im in ims]
    label = transforms.functional.crop(label, i, j, h, w)

    if random.random() > 0.5:
        ims = [transforms.functional.hflip(im) for im in ims]
        label = transforms.functional.hflip(label)

    if random.random() > 0.5:
        ims = [transforms.functional.vflip(im) for im in ims]
        label = transforms.functional.vflip(label)

    return ims, label


def normalize_and_convert_to_tensor(
    ims: List[Image.Image],
    label: Image.Image | None,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the images and label and convert them to PyTorch tensors.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image | None): A PIL Image object representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the normalized
        images and label.
    """
    norm = transforms.Normalize(mean, std)
    ims_tensor = torch.stack([transforms.ToTensor()(im).squeeze() for im in ims])
    _, h, w = ims_tensor.shape
    ims_tensor = ims_tensor.reshape([temporal_size, -1, h, w])  # T*C,H,W -> T,C,H,W
    ims_tensor = torch.stack([norm(im) for im in ims_tensor]).permute(
        [1, 0, 2, 3]
    )  # T,C,H,W -> C,T,H,W
    if label:
        label = torch.from_numpy(np.array(label)).squeeze()
    return ims_tensor, label


def random_crop_flip_and_color_jitter(
    ims: List[Image.Image], 
    label: Image.Image, 
    im_size: int,
    apply_color_jitter: bool = True,
    apply_rotation: bool = True,
    brightness: float = 0.1,
    contrast: float = 0.1, 
    saturation: float = 0.1,
    hue: float = 0.05,
    rotation_degrees: float = 10.0
) -> Tuple[List[Image.Image], Image.Image]:
    """Apply random cropping, flipping, rotation, and color jitter transformations.
    
    This function combines geometric transformations (crop, flip, rotation) with color 
    augmentation that is safe for multispectral data by only applying color jitter to RGB channels.

    Args:
        ims (List[Image.Image]): List of PIL Image objects representing the images.
        label (Image.Image): A PIL Image object representing the label.
        im_size (int): Target size for random crop.
        apply_color_jitter (bool): Whether to apply color jitter augmentation.
        apply_rotation (bool): Whether to apply random rotation.
        brightness (float): Brightness jitter factor.
        contrast (float): Contrast jitter factor.
        saturation (float): Saturation jitter factor.
        hue (float): Hue jitter factor.
        rotation_degrees (float): Range of degrees to randomly rotate.

    Returns:
        Tuple[List[Image.Image], Image.Image]: A tuple containing the transformed list of
        images and label.
    """
    # Apply color jitter first (before geometric transforms to preserve spatial consistency)
    if apply_color_jitter and random.random() > 0.5:
        ims = apply_multispectral_color_jitter(ims, brightness, contrast, saturation, hue)
    
    # Apply rotation before cropping to avoid edge artifacts
    if apply_rotation and random.random() > 0.5:
        angle = random.uniform(-rotation_degrees, rotation_degrees)
        ims = [transforms.functional.rotate(im, angle, expand=False, fill=0) for im in ims]
        if label is not None:
            label = transforms.functional.rotate(label, angle, expand=False, fill=0)
    
    # Apply geometric transformations (crop and flip)
    ims, label = random_crop_and_flip(ims, label, im_size)
    
    return ims, label


def process_and_augment(
    x: np.ndarray,
    y: np.ndarray | None,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    im_size: int = 224,
    augment: bool = True,
    apply_color_jitter: bool = True,
    apply_rotation: bool = True,
    rotation_degrees: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment the given images and labels.

    Args:
        x (np.ndarray): Numpy array representing the images.
        y (np.ndarray): Numpy array representing the label.
        mean (List[float]): The mean of each channel in the image
        std (List[float]): The standard deviation of each channel in the image
        temporal_size: The number of temporal steps
        im_size: Target size for images after augmentation
        augment: Flag to perform augmentations in training mode.
        apply_color_jitter: Flag to apply color jitter augmentation for multispectral data.
        apply_rotation: Flag to apply rotation augmentation.
        rotation_degrees: Range of degrees for random rotation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the processed
        and augmented images and label.
    """
    ims = x.copy()
    label = None
    # convert to PIL for easier transforms
    ims = [Image.fromarray(im) for im in ims]
    if y is not None:
        label = Image.fromarray(y.copy().squeeze())
    if augment:
        ims, label = random_crop_flip_and_color_jitter(
            ims, label, im_size, 
            apply_color_jitter=apply_color_jitter,
            apply_rotation=apply_rotation,
            rotation_degrees=rotation_degrees
        )
    ims, label = normalize_and_convert_to_tensor(ims, label, mean, std, temporal_size)
    return ims, label


def crop_array(
    arr: np.ndarray, left: int, top: int, right: int, bottom: int
) -> np.ndarray:
    """Crop Numpy Image.

    Crop a given array (image) using specified left, top, right, and bottom indices.

    This function supports cropping both grayscale (2D) and color (3D) images.

    Args:
        arr (np.ndarray): The input array (image) to be cropped.
        left (int): The left boundary index for cropping.
        top (int): The top boundary index for cropping.
        right (int): The right boundary index for cropping.
        bottom (int): The bottom boundary index for cropping.

    Returns:
        np.ndarray: The cropped portion of the input array (image).

    Raises:
        ValueError: If the input array is not 2D or 3D.
    """
    if len(arr.shape) == 2:  # Grayscale image (2D array)
        return arr[top:bottom, left:right]
    elif len(arr.shape) == 3:  # Color image (3D array)
        return arr[:, top:bottom, left:right]
    elif len(arr.shape) == 4:  # Color image (3D array)
        return arr[:, :, top:bottom, left:right]
    else:
        raise ValueError("Input array must be a 2D, 3D or 4D array")


def process_test(
    x: np.ndarray,
    y: np.ndarray,
    mean: List[float],
    std: List[float],
    temporal_size: int = 1,
    img_size: int = 512,
    crop_size: int = 224,
    stride: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process and augment test data.

    Args:
        x (np.ndarray): Input image array.
        y (np.ndarray): Corresponding mask array.
        mean (List[float]): Mean values for normalization.
        std: (List[float]): Standard deviation values for normalization.
        temporal_size (int, optional): Temporal dimension size. Defaults to 1.
        img_size (int, optional): Size of the input images. Defaults to
            512.
        crop_size (int, optional): Size of the crops to be extracted from the
            images. Defaults to 224.
        stride (int, optional): Stride for cropping. Defaults to 224.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors containing the processed
            images and masks.
    """
    preprocess_func = partial(
        process_and_augment,
        mean=mean,
        std=std,
        temporal_size=temporal_size,
        augment=False,
    )

    img_crops, mask_crops = [], []
    width, height = img_size, img_size

    for top in range(0, height - crop_size + 1, stride):
        for left in range(0, width - crop_size + 1, stride):
            bottom = top + crop_size
            right = left + crop_size

            img_crops.append(crop_array(x, left, top, right, bottom))
            mask_crops.append(crop_array(y, left, top, right, bottom))

    samples = [preprocess_func(x, y) for x, y in zip(img_crops, mask_crops)]
    imgs = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    return imgs, labels


def get_raster_data(
    fname: str | dict[str, dict[str, str]],
    is_label: bool = True,
    bands: List[int] | None = None,
    no_data_value: int | None = -9999,
    mask_cloud: bool = True,
    water_mask: bool = False,
) -> np.ndarray:
    """Load and process raster data from a file.

    Args:
        fname (str): Filename to load data from.
        is_label (bool): Whether the file is a label file.
        bands (List[int]): Index of bands to select from array.
        no_data_value (int | None): NODATA value in image raster.
        mask_cloud (bool): Perform cloud masking.
        water_mask (bool): Perform water masking.

    Returns:
        np.ndarray: Numpy array representing the processed data.
    """
    if isinstance(fname, dict):
        # @TODO This is used during sliding window inference so masking and processing needs to
        # match what is done to chips in data component
        data = open_mf_tiff_dataset(fname)
        data = data.fillna(no_data_value)
        data = data.band_data.values
    else:
        with rasterio.open(fname) as src:
            data = src.read()
    if (not is_label) and bands:
        data = data[bands, ...]
    return data


def process_data(
    im_fname: str,
    mask_fname: str | None = None,
    no_data_value: int | None = -9999,
    reduce_to_zero: bool = False,
    replace_label: Tuple | None = None,
    bands: List[int] | None = None,
    constant_multiplier: float = 1.0,
    mask_cloud: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process image and mask data from filenames.

    Args:
        im_fname (str): Filename for the image data.
        mask_fname (str | None): Filename for the mask data.
        bands (List[int]): Indices of bands to select from array.
        no_data_value (int | None): NODATA value in image raster.
        reduce_to_zero (bool): Reduces the label index to start from Zero.
        replace_label (Tuple): Tuple of value to replace and the replacement value.
        constant_multiplier (float): Constant multiplier for image.
        mask_cloud (bool): Perform cloud masking.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of numpy arrays representing the processed
        image and mask data.
    """
    arr_x = get_raster_data(
        im_fname,
        is_label=False,
        bands=bands,
        no_data_value=no_data_value,
        mask_cloud=mask_cloud,
        water_mask=False,
    )
    arr_x = arr_x * constant_multiplier
    if mask_fname:
        arr_y = get_raster_data(mask_fname)
        if replace_label:
            arr_y = np.where(arr_y == replace_label[0], replace_label[1], arr_y)
        if reduce_to_zero:
            arr_y -= 1
    else:
        arr_y = None
    return arr_x, arr_y


def load_data_from_csv(fname: str, input_root: str) -> List[Tuple[str, str | None]]:
    """Load data file paths from a CSV file.

    Args:
        fname (str): Filename of the CSV file.
        input_root (str): Root directory for input images and labels.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing file paths for input
        image and label image.
    """
    file_paths = []
    data = pd.read_csv(fname)
    label_present = True if "Label" in data.columns else False
    for _, row in data.iterrows():
        im_path = os.path.join(input_root, row["Input"])
        mask_path = (
            None if not label_present else os.path.join(input_root, row["Label"])
        )
        if os.path.exists(im_path):
            try:
                with rasterio.open(im_path) as src:
                    _ = src.crs
                file_paths.append((im_path, mask_path))
            except Exception as e:
                logging.error(e)
                continue
    return file_paths


class InstaGeoDataset(torch.utils.data.Dataset):
    """InstaGeo PyTorch Dataset for Loading and Handling HLS Data."""

    def __init__(
        self,
        filename: str,
        input_root: str,
        preprocess_func: Callable,
        no_data_value: int | None,
        replace_label: Tuple,
        reduce_to_zero: bool,
        constant_multiplier: float,
        bands: List[int] | None = None,
        include_filenames: bool = False,
        multi_scale_sizes: List[int] | None = None,
        multi_scale_training: bool = False,
    ):
        """Dataset Class for loading and preprocessing the dataset.

        Args:
            filename (str): Filename of the CSV file containing data paths.
            input_root (str): Root directory for input images and labels.
            preprocess_func (Callable): Function to preprocess the data.
            bands (List[int]): Indices of bands to select from array.
            no_data_value (int | None): NODATA value in image raster.
            reduce_to_zero (bool): Reduces the label index to start from Zero.
            replace_label (Tuple): Tuple of value to replace and the replacement value.
            constant_multiplier (float): Constant multiplier for image.
            include_filenames (bool): Flag that determines whether to return filenames.
            multi_scale_sizes (List[int]): List of image sizes for multi-scale training.
            multi_scale_training (bool): Whether to enable multi-scale training.

        """
        self.input_root = input_root
        self.preprocess_func = preprocess_func
        self.bands = bands
        self.file_paths = load_data_from_csv(filename, input_root)
        self.no_data_value = no_data_value
        self.replace_label = replace_label
        self.reduce_to_zero = reduce_to_zero
        self.constant_multiplier = constant_multiplier
        self.include_filenames = include_filenames
        
        # Multi-scale training support
        self.multi_scale_training = multi_scale_training
        self.multi_scale_sizes = multi_scale_sizes or [224, 256, 288, 320]
        self.current_scale_idx = 0

    def set_scale(self, scale_idx: int) -> None:
        """Set the current scale index for multi-scale training.
        
        Args:
            scale_idx (int): Index into the multi_scale_sizes list.
        """
        if self.multi_scale_training and 0 <= scale_idx < len(self.multi_scale_sizes):
            self.current_scale_idx = scale_idx

    def get_current_image_size(self) -> int:
        """Get the current image size for multi-scale training.
        
        Returns:
            int: Current image size.
        """
        if self.multi_scale_training:
            return self.multi_scale_sizes[self.current_scale_idx]
        # Extract size from preprocess_func if it has im_size parameter
        import inspect
        sig = inspect.signature(self.preprocess_func)
        if 'im_size' in sig.parameters:
            return sig.parameters['im_size'].default or 224
        return 224

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sample from dataset.

        Args:
            i (int): Sample index to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors representing the
            processed images and label.
        """
        im_fname, mask_fname = self.file_paths[i]
        arr_x, arr_y = process_data(
            im_fname,
            mask_fname,
            no_data_value=self.no_data_value,
            replace_label=self.replace_label,
            reduce_to_zero=self.reduce_to_zero,
            bands=self.bands,
            constant_multiplier=self.constant_multiplier,
        )
        
        # For multi-scale training, we need to call the preprocess function with dynamic size
        if self.multi_scale_training:
            import inspect
            import random
            from functools import partial
            
            # Randomly select a scale for this sample
            scale_idx = random.randint(0, len(self.multi_scale_sizes) - 1)
            current_size = self.multi_scale_sizes[scale_idx]
            
            # Check if preprocess_func supports im_size parameter
            sig = inspect.signature(self.preprocess_func)
            if 'im_size' in sig.parameters:
                # Create a modified preprocess function with current size
                modified_preprocess = partial(
                    self.preprocess_func.func, 
                    **{**self.preprocess_func.keywords, 'im_size': current_size}
                )
                result = modified_preprocess(arr_x, arr_y)
            else:
                result = self.preprocess_func(arr_x, arr_y)
        else:
            result = self.preprocess_func(arr_x, arr_y)
            
        if self.include_filenames:
            return result, im_fname
        else:
            return result

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_paths)
