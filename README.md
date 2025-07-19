# CIFAR-10 Diffusion Model

🎨 **A diffusion model trained from scratch on CIFAR-10 dataset**

## Model Details
- **Architecture**: SimpleUNet with 16.8M parameters
- **Dataset**: CIFAR-10 (50,000 training images)
- **Training Time**: 14.54 minutes on RTX 3060
- **Final Loss**: 0.0363
- **Image Size**: 32x32 RGB
- **Framework**: PyTorch

## Quick Start

```python
import torch
from model import SimpleUNet, DDPMScheduler, generate_images

# Load the trained model
checkpoint = torch.load('complete_diffusion_model.pth')
model = SimpleUNet(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize scheduler
scheduler = DDPMScheduler(**checkpoint['diffusion_config'])

# Generate images
generated_images = generate_images(model, scheduler, num_images=8)
```

## Installation

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 matplotlib tqdm pillow numpy
```

## Files Included
- `complete_diffusion_model.pth` - Complete model with config (64MB)
- `model_info.json` - Training details and metadata
- `diffusion_model_final.pth` - Final training checkpoint (64MB)
- `inference_example.py` - Ready-to-use inference script

## Training Details
- **Epochs**: 20
- **Batch Size**: 128
- **Learning Rate**: 1e-4 (CosineAnnealingLR)
- **Optimizer**: AdamW
- **GPU**: NVIDIA RTX 3060 (0.43GB VRAM used)
- **Loss Reduction**: 73% (from 0.1349 to 0.0363)

## Hardware Requirements
- **Minimum**: 1GB VRAM for inference
- **Recommended**: 2GB+ VRAM for training extensions
- **CPU**: Works but slower

## Results
The model generates colorful abstract patterns that capture CIFAR-10's color distributions. 
With more training epochs (50-100), it should produce more recognizable objects.

## Improvements
To get better results:
1. **Train longer**: 50-100 epochs instead of 20
2. **Larger model**: Increase channels/layers
3. **Advanced sampling**: DDIM, DPM-Solver
4. **Better datasets**: CelebA, ImageNet
5. **Learning rate**: Experiment with schedules

## Model Architecture
- **U-Net based** with ResNet blocks
- **Time embedding** for diffusion timesteps
- **Attention layers** at multiple resolutions
- **Skip connections** for better gradient flow

## Citation
```bibtex
@misc{cifar10-diffusion-2025,
  title={CIFAR-10 Diffusion Model},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/cifar10-diffusion}
}
```

## License
MIT License - Feel free to use and modify!

---
**Created**: July 19, 2025  
**Training Time**: 14.54 minutes  
**GPU**: NVIDIA RTX 3060  
**Framework**: PyTorch
