"""
Attention Map Visualization for MedViT

This script visualizes attention maps from trained MedViT models.
It captures attention weights from:
- LFP blocks: Neighborhood Attention or Standard Multi-Head Attention
- GFP blocks: E-MHSA (Efficient Multi-Head Self Attention)

Usage:
    python src/visualize_attention.py \
        --checkpoint_path ./checkpoint/MedViT_tiny_brain_tumor.pth \
        --image_path ./sample_image.jpg \
        --model_name MedViT_tiny \
        --num_classes 4
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from models.MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
from models.MedViT import LFP, GFP, E_MHSA, StandardMultiHeadAttention


class AttentionExtractor:
    """Extract attention maps from MedViT model using forward hooks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""

        def get_e_mhsa_hook(name: str):
            """Hook for E_MHSA attention in GFP blocks."""
            def hook(module, input, output):
                # We need to modify E_MHSA to store attention weights
                # For now, capture the input/output shapes
                self.feature_maps[name] = {
                    'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                    'output_shape': output.shape
                }
            return hook

        def get_standard_attn_hook(name: str):
            """Hook for StandardMultiHeadAttention in LFP blocks."""
            def hook(module, input, output):
                self.feature_maps[name] = {
                    'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                    'output_shape': output.shape
                }
            return hook

        # Register hooks for each attention module
        for name, module in self.model.named_modules():
            if isinstance(module, E_MHSA):
                hook = module.register_forward_hook(get_e_mhsa_hook(name))
                self.hooks.append(hook)
            elif isinstance(module, StandardMultiHeadAttention):
                hook = module.register_forward_hook(get_standard_attn_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self):
        """Clear stored attention maps."""
        self.attention_maps.clear()
        self.feature_maps.clear()


class AttentionVisualizer:
    """Visualize attention maps from MedViT model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        img_size: int = 224
    ):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Wrap attention modules to capture weights
        self._wrap_attention_modules()

    def _wrap_attention_modules(self):
        """Wrap attention modules to capture attention weights during forward pass."""

        for name, module in self.model.named_modules():
            if isinstance(module, E_MHSA):
                # Wrap E_MHSA forward to capture attention
                original_forward = module.forward
                module._attn_name = name
                module._visualizer = self

                def make_e_mhsa_forward(mod, orig_forward):
                    def new_forward(x):
                        B, N, C = x.shape
                        q = mod.q(x)
                        q = q.reshape(B, N, mod.num_heads, int(C // mod.num_heads)).permute(0, 2, 1, 3)

                        if mod.sr_ratio > 1:
                            x_ = x.transpose(1, 2)
                            x_ = mod.sr(x_)
                            if not torch.onnx.is_in_onnx_export() and not mod.is_bn_merged:
                                x_ = mod.norm(x_)
                            x_ = x_.transpose(1, 2)
                            k = mod.k(x_)
                            k = k.reshape(B, -1, mod.num_heads, int(C // mod.num_heads)).permute(0, 2, 3, 1)
                            v = mod.v(x_)
                            v = v.reshape(B, -1, mod.num_heads, int(C // mod.num_heads)).permute(0, 2, 1, 3)
                        else:
                            k = mod.k(x)
                            k = k.reshape(B, -1, mod.num_heads, int(C // mod.num_heads)).permute(0, 2, 3, 1)
                            v = mod.v(x)
                            v = v.reshape(B, -1, mod.num_heads, int(C // mod.num_heads)).permute(0, 2, 1, 3)

                        attn = (q @ k) * mod.scale
                        attn = attn.softmax(dim=-1)

                        # Store attention weights
                        mod._visualizer.attention_weights[mod._attn_name] = attn.detach().cpu()

                        attn = mod.attn_drop(attn)
                        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                        x = mod.proj(x)
                        x = mod.proj_drop(x)
                        return x
                    return new_forward

                module.forward = make_e_mhsa_forward(module, original_forward)

            elif isinstance(module, StandardMultiHeadAttention):
                # Wrap StandardMultiHeadAttention forward to capture attention
                module._attn_name = name
                module._visualizer = self

                def make_std_attn_forward(mod):
                    def new_forward(x):
                        B, H, W, C = x.shape
                        N = H * W
                        x = x.view(B, N, C)

                        qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, mod.head_dim).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]

                        attn = (q @ k.transpose(-2, -1)) * mod.scale
                        attn = attn.softmax(dim=-1)

                        # Store attention weights
                        mod._visualizer.attention_weights[mod._attn_name] = attn.detach().cpu()

                        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                        x = mod.proj(x)
                        x = mod.proj_drop(x)
                        x = x.view(B, H, W, C)
                        return x
                    return new_forward

                module.forward = make_std_attn_forward(module)

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load and preprocess image."""
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img.resize((self.img_size, self.img_size)))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor, img_np

    def get_attention_maps(self, image_path: str) -> Dict[str, torch.Tensor]:
        """Extract attention maps for a given image."""
        self.attention_weights.clear()

        img_tensor, img_np = self.preprocess_image(image_path)

        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_class = output.argmax(dim=1).item()

        return self.attention_weights.copy(), img_np, pred_class

    def aggregate_attention(
        self,
        attn: torch.Tensor,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Aggregate multi-head attention into a single map.

        Args:
            attn: Attention tensor of shape (B, num_heads, N, N) or (B, num_heads, N, M)
            method: Aggregation method ('mean', 'max', 'cls')

        Returns:
            Aggregated attention map
        """
        # Remove batch dimension
        attn = attn[0]  # (num_heads, N, N) or (num_heads, N, M)

        if method == 'mean':
            # Average over heads and query positions
            attn_map = attn.mean(dim=0).mean(dim=0)  # (N,) or (M,)
        elif method == 'max':
            # Max over heads, mean over query positions
            attn_map = attn.max(dim=0)[0].mean(dim=0)
        elif method == 'cls':
            # Use first token (CLS-like) attention
            attn_map = attn.mean(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return attn_map.numpy()

    def reshape_attention_to_image(
        self,
        attn_map: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Reshape 1D attention map to 2D image space."""
        # Try to find the spatial size
        n = len(attn_map)
        h = w = int(np.sqrt(n))

        if h * w != n:
            # Handle non-square attention maps
            # Try common aspect ratios
            for aspect in [1, 2, 4, 7, 8, 14]:
                if n % aspect == 0:
                    h = aspect
                    w = n // aspect
                    if abs(h - w) < abs(aspect - n // aspect):
                        break

        try:
            attn_2d = attn_map.reshape(h, w)
        except ValueError:
            # Fallback: use closest square
            side = int(np.ceil(np.sqrt(n)))
            padded = np.zeros(side * side)
            padded[:n] = attn_map
            attn_2d = padded.reshape(side, side)

        # Resize to target size
        from PIL import Image
        attn_img = Image.fromarray(attn_2d)
        attn_img = attn_img.resize(target_size, Image.BILINEAR)
        return np.array(attn_img)

    def visualize(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
        layer_indices: Optional[List[int]] = None,
        cmap: str = 'jet',
        alpha: float = 0.5
    ):
        """
        Visualize attention maps overlaid on the input image.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            show: Whether to display the plot
            layer_indices: Which layers to visualize (None = all)
            cmap: Colormap for attention visualization
            alpha: Transparency of attention overlay
        """
        attention_maps, img_np, pred_class = self.get_attention_maps(image_path)

        if not attention_maps:
            print("No attention maps captured. Make sure the model uses supported attention modules.")
            return

        # Filter layers if specified
        layer_names = list(attention_maps.keys())
        if layer_indices is not None:
            layer_names = [layer_names[i] for i in layer_indices if i < len(layer_names)]

        n_layers = len(layer_names)
        if n_layers == 0:
            print("No layers to visualize.")
            return

        # Create figure
        n_cols = min(4, n_layers + 1)
        n_rows = (n_layers + 1 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Original (Pred: {pred_class})')
        axes[0, 0].axis('off')

        # Plot attention maps
        for idx, layer_name in enumerate(layer_names):
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols

            attn = attention_maps[layer_name]

            # Aggregate attention
            attn_map = self.aggregate_attention(attn, method='mean')

            # Reshape to image size
            attn_2d = self.reshape_attention_to_image(
                attn_map,
                (self.img_size, self.img_size)
            )

            # Normalize
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

            # Plot
            ax = axes[row, col]
            ax.imshow(img_np)
            im = ax.imshow(attn_2d, cmap=cmap, alpha=alpha)

            # Shorten layer name for display
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            if len(layer_name) > 30:
                short_name = f"...{layer_name[-25:]}"
            ax.set_title(f'{short_name}', fontsize=8)
            ax.axis('off')

        # Hide unused axes
        for idx in range(n_layers + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return attention_maps, pred_class

    def visualize_all_heads(
        self,
        image_path: str,
        layer_name: str,
        output_path: Optional[str] = None,
        show: bool = True,
        cmap: str = 'jet',
        alpha: float = 0.5
    ):
        """
        Visualize attention from all heads for a specific layer.

        Args:
            image_path: Path to input image
            layer_name: Name of the layer to visualize
            output_path: Path to save visualization
            show: Whether to display the plot
            cmap: Colormap for attention visualization
            alpha: Transparency of attention overlay
        """
        attention_maps, img_np, pred_class = self.get_attention_maps(image_path)

        if layer_name not in attention_maps:
            available = list(attention_maps.keys())
            print(f"Layer '{layer_name}' not found. Available layers: {available}")
            return

        attn = attention_maps[layer_name][0]  # Remove batch dim: (num_heads, N, N)
        num_heads = attn.shape[0]

        # Create figure
        n_cols = min(4, num_heads + 1)
        n_rows = (num_heads + 1 + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Original (Pred: {pred_class})')
        axes[0, 0].axis('off')

        # Plot each head
        for head_idx in range(num_heads):
            row = (head_idx + 1) // n_cols
            col = (head_idx + 1) % n_cols

            # Get attention for this head, average over query positions
            head_attn = attn[head_idx].mean(dim=0).numpy()  # (N,)

            # Reshape to image size
            attn_2d = self.reshape_attention_to_image(
                head_attn,
                (self.img_size, self.img_size)
            )

            # Normalize
            attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

            ax = axes[row, col]
            ax.imshow(img_np)
            ax.imshow(attn_2d, cmap=cmap, alpha=alpha)
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')

        # Hide unused axes
        for idx in range(num_heads + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f'Attention Heads: {layer_name}', fontsize=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")

        if show:
            plt.show()
        else:
            plt.close()


def load_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    """Load MedViT model from checkpoint."""

    model_dict = {
        'MedViT_tiny': MedViT_tiny,
        'MedViT_small': MedViT_small,
        'MedViT_base': MedViT_base,
        'MedViT_large': MedViT_large,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")

    model = model_dict[model_name](num_classes=num_classes)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load with strict=False to handle potential mismatches
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")

    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize MedViT attention maps')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/MedViT_tiny.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny',
                        choices=['MedViT_tiny', 'MedViT_small', 'MedViT_base', 'MedViT_large'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of output classes')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Display visualization')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='Do not display visualization')
    parser.add_argument('--cmap', type=str, default='jet',
                        help='Colormap for attention overlay')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Transparency of attention overlay')

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        args.model_name,
        args.checkpoint_path,
        args.num_classes,
        device
    )

    # Create visualizer
    visualizer = AttentionVisualizer(model, device)

    # Visualize
    visualizer.visualize(
        args.image_path,
        output_path=args.output_path,
        show=args.show,
        cmap=args.cmap,
        alpha=args.alpha
    )


if __name__ == '__main__':
    main()
