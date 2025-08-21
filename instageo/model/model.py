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

"""Model Module."""

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import requests  # type: ignore
import torch
import torch.nn as nn
import yaml  # type: ignore
from absl import logging

from instageo.model.Prithvi import ViTEncoder, get_3d_sincos_pos_embed


def download_file(url: str, filename: str | Path, retries: int = 3) -> None:
    """Downloads a file from the given URL and saves it to a local file.

    Args:
        url (str): The URL from which to download the file.
        filename (str): The local path where the file will be saved.
        retries (int, optional): The number of times to retry the download
                                 in case of failure. Defaults to 3.

    Raises:
        Exception: If the download fails after the specified number of retries.

    Returns:
        None
    """
    if os.path.exists(filename):
        logging.info(f"File '{filename}' already exists. Skipping download.")
        return

    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logging.info(f"Download successful on attempt {attempt + 1}")
                break
            else:
                logging.warning(
                    f"Attempt {attempt + 1} failed with status code {response.status_code}"  # noqa
                )
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < retries - 1:
            time.sleep(2)

    else:
        raise Exception("Failed to download the file after several attempts.")


class Norm2D(nn.Module):
    """A normalization layer for 2D inputs.

    This class implements a 2D normalization layer using Layer Normalization.
    It is designed to normalize 2D inputs (e.g., images or feature maps in a
    convolutional neural network).

    Attributes:
        ln (nn.LayerNorm): The layer normalization component.

    Args:
        embed_dim (int): The number of features of the input tensor (i.e., the number of
            channels in the case of images).

    Methods:
        forward: Applies normalization to the input tensor.
    """

    def __init__(self, embed_dim: int):
        """Initializes the Norm2D module.

        Args:
            embed_dim (int): The number of features of the input tensor.
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the normalization process to the input tensor.

        Args:
            x (torch.Tensor): A 4D input tensor with shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The normalized tensor, having the same shape as the input.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PrithviSeg(nn.Module):
    """Prithvi Segmentation Model."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi configuration and weights and sets up a ViTEncoder backbone
        along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
        """
        super().__init__()
        weights_dir = Path.home() / ".instageo" / "prithvi"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "Prithvi_EO_V1_100M.pt"
        cfg_path = weights_dir / "config.yaml"
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt?download=true",  # noqa
            weights_path,
        )
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/config.yaml",  # noqa
            cfg_path,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]

        model_args["num_frames"] = temporal_step
        model_args["img_size"] = image_size
        self.model_args = model_args
        # instantiate model
        model = ViTEncoder(**model_args)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        filtered_checkpoint_state_dict = {
            key[len("encoder.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("encoder.")
        }
        filtered_checkpoint_state_dict["pos_embed"] = (
            torch.from_numpy(
                get_3d_sincos_pos_embed(
                    768,
                    (temporal_step, image_size // 16, image_size // 16),
                    cls_token=True,
                )
            )
            .float()
            .unsqueeze(0)
        )
        _ = model.load_state_dict(filtered_checkpoint_state_dict)

        self.prithvi_100M_backbone = model

        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            """Upscaling block.

            Args:
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.

            Returns:
                An upscaling block configured to upscale spatially.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        embed_dims = [
            (model_args["embed_dim"] * model_args["num_frames"]) // (2**i)
            for i in range(5)
        ]
        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
            nn.Conv2d(
                kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes
            ),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        features = self.prithvi_100M_backbone(img)
        # drop cls token
        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int(
            np.sqrt(reshaped_features.shape[1] // self.model_args["num_frames"])
        )
        reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
            features.shape[0], -1, feature_img_side_length, feature_img_side_length
        )

        out = self.segmentation_head(reshaped_features)
        return out


class AttentionBlock(nn.Module):
    """Simple channel attention block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip_connection(residual)
        out = self.relu(out)
        return out


class ImprovedPrithviSeg(nn.Module):
    """Improved Prithvi Segmentation Model with skip connections and attention."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        use_attention: bool = True,
        use_skip_connections: bool = True,
    ) -> None:
        """Initialize the ImprovedPrithviSeg model.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            use_attention (bool): Whether to use attention mechanisms.
            use_skip_connections (bool): Whether to use skip connections.
        """
        super().__init__()
        
        # Initialize base Prithvi model
        self.base_model = PrithviSeg(
            temporal_step=temporal_step,
            image_size=image_size,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )
        
        self.use_attention = use_attention
        self.use_skip_connections = use_skip_connections
        
        # Get embedding dimensions from the base model
        model_args = self.base_model.model_args
        embed_dims = [
            (model_args["embed_dim"] * model_args["num_frames"]) // (2**i)
            for i in range(5)
        ]
        
        # Enhanced segmentation head with skip connections and attention
        if use_skip_connections or use_attention:
            self.enhanced_head = self._build_enhanced_head(embed_dims, num_classes)
        else:
            self.enhanced_head = None
    
    def _build_enhanced_head(self, embed_dims: List[int], num_classes: int) -> nn.Module:
        """Build enhanced segmentation head with skip connections and attention."""
        layers = []
        
        for i in range(4):
            in_dim = embed_dims[i]
            out_dim = embed_dims[i + 1]
            
            # Upscaling block
            upscale = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
            )
            
            # Add residual block if using skip connections
            if self.use_skip_connections:
                residual = ResidualBlock(out_dim, out_dim)
                block = nn.Sequential(upscale, residual)
            else:
                block = upscale
            
            # Add attention if enabled
            if self.use_attention:
                attention = AttentionBlock(out_dim)
                block = nn.Sequential(block, attention)
            
            layers.append(block)
        
        # Final classification layer
        layers.append(
            nn.Conv2d(
                kernel_size=1, 
                in_channels=embed_dims[-1], 
                out_channels=num_classes
            )
        )
        
        return nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        if self.enhanced_head is not None:
            # Use enhanced head
            features = self.base_model.prithvi_100M_backbone(img)
            # drop cls token
            reshaped_features = features[:, 1:, :]
            feature_img_side_length = int(
                np.sqrt(reshaped_features.shape[1] // self.base_model.model_args["num_frames"])
            )
            reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
                features.shape[0], -1, feature_img_side_length, feature_img_side_length
            )
            out = self.enhanced_head(reshaped_features)
            return out
        else:
            # Use base model
            return self.base_model(img)
    """Prithvi Segmentation Model."""

    def __init__(
        self,
        temporal_step: int = 1,
        image_size: int = 224,
        num_classes: int = 2,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialize the PrithviSeg model.

        This model is designed for image segmentation tasks on remote sensing data.
        It loads Prithvi configuration and weights and sets up a ViTEncoder backbone
        along with a segmentation head.

        Args:
            temporal_step (int): Size of temporal dimension.
            image_size (int): Size of input image.
            num_classes (int): Number of target classes.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
        """
        super().__init__()
        weights_dir = Path.home() / ".instageo" / "prithvi"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "Prithvi_EO_V1_100M.pt"
        cfg_path = weights_dir / "config.yaml"
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt?download=true",  # noqa
            weights_path,
        )
        download_file(
            "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/raw/main/config.yaml",  # noqa
            cfg_path,
        )
        checkpoint = torch.load(weights_path, map_location="cpu")
        with open(cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]

        model_args["num_frames"] = temporal_step
        model_args["img_size"] = image_size
        self.model_args = model_args
        # instantiate model
        model = ViTEncoder(**model_args)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        filtered_checkpoint_state_dict = {
            key[len("encoder.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("encoder.")
        }
        filtered_checkpoint_state_dict["pos_embed"] = (
            torch.from_numpy(
                get_3d_sincos_pos_embed(
                    768,
                    (temporal_step, image_size // 16, image_size // 16),
                    cls_token=True,
                )
            )
            .float()
            .unsqueeze(0)
        )
        _ = model.load_state_dict(filtered_checkpoint_state_dict)

        self.prithvi_100M_backbone = model

        def upscaling_block(in_channels: int, out_channels: int) -> nn.Module:
            """Upscaling block.

            Args:
                in_channels (int): number of input channels.
                out_channels (int): number of output channels.

            Returns:
                An upscaling block configured to upscale spatially.
            """
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        embed_dims = [
            (model_args["embed_dim"] * model_args["num_frames"]) // (2**i)
            for i in range(5)
        ]
        self.segmentation_head = nn.Sequential(
            *[upscaling_block(embed_dims[i], embed_dims[i + 1]) for i in range(4)],
            nn.Conv2d(
                kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes
            ),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            img (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: Output tensor after image segmentation.
        """
        features = self.prithvi_100M_backbone(img)
        # drop cls token
        reshaped_features = features[:, 1:, :]
        feature_img_side_length = int(
            np.sqrt(reshaped_features.shape[1] // self.model_args["num_frames"])
        )
        reshaped_features = reshaped_features.permute(0, 2, 1).reshape(
            features.shape[0], -1, feature_img_side_length, feature_img_side_length
        )

        out = self.segmentation_head(reshaped_features)
        return out
