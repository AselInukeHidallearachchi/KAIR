import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from utils import utils_image as util
import requests
from pathlib import Path

MODEL_URL = "https://github.com/xinntao/KAIR/releases/download/v1.0/ffdnet_color.pth"

class FFDNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R'):
        super(FFDNet, self).__init__()
        self.nc = nc

        model = []
        
        # First layer
        model.append(nn.Conv2d(in_nc*4+1, nc, 3, padding=1, bias=True))
        model.append(nn.ReLU(inplace=True))
        
        # Body
        for i in range(nb-2):
            model.append(nn.Conv2d(nc, nc, 3, padding=1, bias=True))
            model.append(nn.ReLU(inplace=True))
            
        model.append(nn.Conv2d(nc, out_nc*4, 3, padding=1, bias=True))
        
        self.model = nn.Sequential(*model)

    def forward(self, x, sigma=None):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2 - h)
        paddingRight = int(np.ceil(w/2)*2 - w)
        x = torch.nn.functional.pad(x, [0, paddingRight, 0, paddingBottom], mode='reflect')

        if sigma is None:
            sigma = torch.zeros((x.size()[0], 1, x.size()[2]//2, x.size()[3]//2)).type_as(x)
        else:
            sigma = sigma.view(1, 1, 1, 1).expand((1, 1, x.size()[2]//2, x.size()[3]//2))

        # Downscale
        x = torch.nn.functional.pixel_unshuffle(x, 2)
        
        # Concatenate noise level
        x = torch.cat([x, sigma], dim=1)
        
        # Process
        x = self.model(x)
        
        # Upscale
        x = nn.functional.pixel_shuffle(x, 2)
        x = x[..., :h, :w]
        
        return x

def download_model(model_path):
    """Download the pre-trained model if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading pre-trained model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Download completed!")

def main():
    parser = argparse.ArgumentParser(description='FFDNet Image Denoising')
    parser.add_argument('--input', type=str, default='./noisy.png', help='Path to noisy image')
    parser.add_argument('--output', type=str, default='./denoised.png', help='Path to save denoised image')
    parser.add_argument('--noise_level', type=float, default=15, help='Noise level (sigma)')
    parser.add_argument('--model_path', type=str, default='./model_zoo/ffdnet/ffdnet_color.pth', help='Path to model')
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Ensure model exists
    try:
        download_model(args.model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {str(e)}")

    # Load model
    try:
    model = FFDNet(in_nc=3, out_nc=3, nc=96)
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    # Read and preprocess image
    try:
        img_noisy = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_noisy is None:
            raise ValueError(f"Could not read input image: {args.input}")
        
        # Convert BGR to RGB
    img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB)
    img_noisy = img_noisy.astype(np.float32) / 255.0
    img_noisy = util.single2tensor4(img_noisy).to(device)
    except Exception as e:
        raise RuntimeError(f"Error processing input image: {str(e)}")

    # Prepare noise level
    noise_level = torch.FloatTensor([args.noise_level/255.]).to(device)

    # Denoise image
    try:
    with torch.no_grad():
            img_denoised = model(img_noisy, noise_level)
    except Exception as e:
        raise RuntimeError(f"Error during denoising: {str(e)}")

    # Post-process and save
    try:
    img_denoised = util.tensor2single(img_denoised)
    img_denoised = np.clip(img_denoised * 255.0, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR for saving
    img_denoised = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2BGR)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # Save the image
    cv2.imwrite(args.output, img_denoised)
    print(f'Denoised image saved to {args.output}')
    except Exception as e:
        raise RuntimeError(f"Error saving output image: {str(e)}")

    # Calculate PSNR if clean image exists
    try:
        clean_path = args.input.replace('noisy', 'clean')
        if os.path.exists(clean_path):
            img_clean = cv2.imread(clean_path)
        if img_clean is not None:
            img_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
            psnr = util.calculate_psnr(img_denoised, img_clean)
            print(f'PSNR: {psnr:.2f} dB')
    except Exception as e:
        print(f"Warning: Could not calculate PSNR: {str(e)}")

if __name__ == '__main__':
    try:
    main()
    except Exception as e:
        print(f"Error: {str(e)}")
