from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from instageo.model.dataloader import (
    InstaGeoDataset,
    apply_multispectral_color_jitter,
    crop_array,
    get_raster_data,
    load_data_from_csv,
    normalize_and_convert_to_tensor,
    process_and_augment,
    process_data,
    process_test,
    random_crop_and_flip,
    random_crop_flip_and_color_jitter,
)


def test_random_crop_and_flip():
    # Create dummy images and label
    ims = [Image.new("L", (256, 256)) for _ in range(3)]
    label = Image.new("L", (256, 256))

    # Apply function
    transformed_ims, transformed_label = random_crop_and_flip(ims, label, im_size=224)

    # Check output types and dimensions
    assert isinstance(transformed_ims, list)
    assert isinstance(transformed_label, Image.Image)
    assert all(isinstance(im, Image.Image) for im in transformed_ims)
    assert all(im.size == (224, 224) for im in transformed_ims)
    assert transformed_label.size == (224, 224)


def test_normalize_and_convert_to_tensor():
    # Create dummy images and label
    ims = [Image.new("L", (224, 224)) for _ in range(3)]
    label = Image.new("L", (224, 224))

    tensor_ims, tensor_label = normalize_and_convert_to_tensor(
        ims, label, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    )

    assert isinstance(tensor_ims, torch.Tensor)
    assert isinstance(tensor_label, torch.Tensor)
    assert tensor_ims.shape == torch.Size([3, 1, 224, 224])
    assert tensor_label.shape == torch.Size([224, 224])


def test_process_and_augment():
    x = np.random.rand(6, 256, 256)
    y = np.random.rand(256, 256)
    processed_ims, processed_label = process_and_augment(
        x, y, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], temporal_size=2, im_size=224
    )

    assert processed_ims.shape == torch.Size([3, 2, 224, 224])
    assert processed_label.shape == torch.Size([224, 224])


def test_get_raster_data_with_str():
    test_fname = "tests/data/sample.tif"
    result = get_raster_data(test_fname, is_label=False, bands=[0])
    assert isinstance(result, np.ndarray)


def test_get_raster_data_with_dict():
    band_files = {
        "tiles": {
            "band1": "tests/data/sample.tif",
            "band2": "tests/data/sample.tif",
        },
        "fmasks": {
            "band1": "tests/data/fmask.tif",
            "band2": "tests/data/fmask.tif",
        },
    }
    result = get_raster_data(band_files, mask_cloud=True)
    assert isinstance(result, np.ndarray)


def test_process_data():
    im_test_fname = "tests/data/sample.tif"
    mask_test_fname = "tests/data/sample.tif"

    arr_x, arr_y = process_data(
        im_test_fname,
        mask_test_fname,
        no_data_value=-1,
        replace_label=(-1, 0),
        reduce_to_zero=True,
    )

    assert isinstance(arr_x, np.ndarray)
    assert isinstance(arr_y, np.ndarray)
    assert arr_x.shape[:-2] == arr_y.shape[:-2]


def test_process_data_without_label():
    im_test_fname = "tests/data/sample.tif"

    arr_x, arr_y = process_data(
        im_test_fname, no_data_value=-1, replace_label=(-1, 0), reduce_to_zero=True
    )

    assert isinstance(arr_x, np.ndarray)
    assert arr_y is None


def test_crop_2d_array():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cropped = crop_array(arr, 1, 1, 3, 3)
    expected = np.array([[5, 6], [8, 9]])
    assert np.array_equal(cropped, expected)


def test_crop_3d_array():
    arr = np.zeros((3, 3, 3))  # Example 3D array
    arr[:, 1, 1] = 1  # Modify the array for testing
    cropped = crop_array(arr, 1, 1, 3, 3)
    expected = np.zeros((3, 2, 2))
    expected[:, 0, 0] = 1
    assert np.array_equal(cropped, expected)


def test_invalid_dimensions():
    arr = np.array([1, 2, 3])  # 1D array, not valid
    with pytest.raises(ValueError):
        crop_array(arr, 0, 0, 1, 1)


def test_boundary_conditions():
    arr = np.array([[1, 2], [3, 4]])
    cropped = crop_array(arr, 0, 0, 2, 2)
    assert np.array_equal(cropped, arr)


def test_non_integer_indices():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        crop_array(arr, 0.5, 0.5, 1.5, 1.5)


def test_output_types_and_shapes():
    x = np.random.rand(3, 512, 512)
    y = np.random.rand(512, 512)
    mean = [0.5, 0.5, 0.5]
    std = [0.1, 0.1, 0.1]
    imgs, labels = process_test(x, y, mean, std)
    assert isinstance(imgs, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert imgs.shape == torch.Size([4, 3, 1, 224, 224])
    assert labels.shape == torch.Size([4, 224, 224])


def test_invalid_inputs():
    x = np.random.rand(3, 512, 512)
    y = np.random.rand(512, 512)
    mean = [0.5, 0.5, "invalid"]
    std = [0.1, 0.1, 0.1]
    with pytest.raises(Exception):
        process_test(x, y, mean, std)


def test_load_data_from_csv():
    sample_filename = "/tmp/sample_data.csv"
    pd.DataFrame(
        {
            "Input": ["sample.tif", "sample.tif"],
            "Label": ["sample.tif", "sample.tif"],
        }
    ).to_csv(sample_filename)
    data = load_data_from_csv(sample_filename, input_root="tests/data")
    assert data == [
        ("tests/data/sample.tif", "tests/data/sample.tif"),
        ("tests/data/sample.tif", "tests/data/sample.tif"),
    ]


def test_instageo_dataset():
    sample_filename = "/tmp/sample_data.csv"
    pd.DataFrame(
        {
            "Input": ["sample.tif", "sample.tif"],
            "Label": ["sample.tif", "sample.tif"],
        }
    ).to_csv(sample_filename)
    dataset = InstaGeoDataset(
        filename=sample_filename,
        input_root="tests/data",
        preprocess_func=partial(
            process_and_augment,
            mean=[0.0],
            std=[1.0],
            temporal_size=1,
            im_size=224,
        ),
        bands=[0],
        replace_label=None,
        reduce_to_zero=False,
        no_data_value=-1,
        constant_multiplier=0.001,
    )
    im, label = next(iter(dataset))
    assert isinstance(im, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert im.shape == torch.Size([1, 1, 224, 224])
    assert label.shape == torch.Size([224, 224])


def test_apply_multispectral_color_jitter():
    """Test that color jitter is applied only to RGB channels."""
    # Create 6-channel multispectral data (simulating Blue, Green, Red, NIR, SWIR1, SWIR2)
    ims = [Image.new("L", (256, 256), color=50 + i * 10) for i in range(6)]
    
    # Store original pixel values for comparison
    original_values = [np.array(im) for im in ims]
    
    # Apply color jitter
    transformed_ims = apply_multispectral_color_jitter(ims)
    
    # Check that we get the same number of images back
    assert len(transformed_ims) == 6
    assert all(isinstance(im, Image.Image) for im in transformed_ims)
    
    # Check that all images have the same size
    assert all(im.size == (256, 256) for im in transformed_ims)
    
    # Convert back to numpy for detailed checking
    transformed_values = [np.array(im) for im in transformed_ims]
    
    # Check that NIR, SWIR1, SWIR2 channels (indices 3, 4, 5) are unchanged
    for i in range(3, 6):
        np.testing.assert_array_equal(
            original_values[i], 
            transformed_values[i],
            err_msg=f"Channel {i} should be unchanged but was modified"
        )


def test_apply_multispectral_color_jitter_with_few_channels():
    """Test color jitter behavior with fewer than 3 channels."""
    # Create 2-channel data
    ims = [Image.new("L", (256, 256), color=50 + i * 10) for i in range(2)]
    
    # Apply color jitter - should apply to all channels when < 3 channels
    transformed_ims = apply_multispectral_color_jitter(ims)
    
    assert len(transformed_ims) == 2
    assert all(isinstance(im, Image.Image) for im in transformed_ims)
    assert all(im.size == (256, 256) for im in transformed_ims)


def test_random_crop_flip_and_color_jitter():
    """Test the combined augmentation function."""
    # Create 6-channel multispectral data
    ims = [Image.new("L", (256, 256), color=50 + i * 10) for i in range(6)]
    label = Image.new("L", (256, 256), color=128)
    
    # Apply combined transformations
    transformed_ims, transformed_label = random_crop_flip_and_color_jitter(
        ims, label, im_size=224, apply_color_jitter=True
    )
    
    # Check output types and dimensions
    assert isinstance(transformed_ims, list)
    assert isinstance(transformed_label, Image.Image)
    assert len(transformed_ims) == 6
    assert all(isinstance(im, Image.Image) for im in transformed_ims)
    assert all(im.size == (224, 224) for im in transformed_ims)
    assert transformed_label.size == (224, 224)


def test_random_crop_flip_and_color_jitter_no_color():
    """Test the combined augmentation function without color jitter."""
    # Create 6-channel multispectral data
    ims = [Image.new("L", (256, 256), color=50 + i * 10) for i in range(6)]
    label = Image.new("L", (256, 256), color=128)
    
    # Store original values for the non-visible channels
    original_values = [np.array(im) for im in ims[3:]]  # NIR, SWIR1, SWIR2
    
    # Apply transformations without color jitter
    transformed_ims, transformed_label = random_crop_flip_and_color_jitter(
        ims, label, im_size=224, apply_color_jitter=False
    )
    
    # Check output dimensions
    assert len(transformed_ims) == 6
    assert all(im.size == (224, 224) for im in transformed_ims)
    assert transformed_label.size == (224, 224)


def test_process_and_augment_with_color_jitter():
    """Test process_and_augment with color jitter enabled."""
    x = np.random.rand(6, 256, 256).astype(np.float32) * 255
    x = x.astype(np.uint8)  # Convert to uint8 for PIL Image compatibility
    y = np.random.rand(256, 256).astype(np.float32) * 255
    y = y.astype(np.uint8)
    
    processed_ims, processed_label = process_and_augment(
        x, y, 
        mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
        temporal_size=1, 
        im_size=224,
        augment=True,
        apply_color_jitter=True
    )

    assert processed_ims.shape == torch.Size([6, 1, 224, 224])
    assert processed_label.shape == torch.Size([224, 224])


def test_process_and_augment_without_color_jitter():
    """Test process_and_augment with color jitter disabled."""
    x = np.random.rand(6, 256, 256).astype(np.float32) * 255
    x = x.astype(np.uint8)  # Convert to uint8 for PIL Image compatibility
    y = np.random.rand(256, 256).astype(np.float32) * 255
    y = y.astype(np.uint8)
    
    processed_ims, processed_label = process_and_augment(
        x, y, 
        mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
        temporal_size=1, 
        im_size=224,
        augment=True,
        apply_color_jitter=False
    )

    assert processed_ims.shape == torch.Size([6, 1, 224, 224])
    assert processed_label.shape == torch.Size([224, 224])
