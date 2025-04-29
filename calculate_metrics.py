import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import math

def calculate_metrics(img1, img2):
    """Calculate PSNR and SSIM between two images."""
    # Convert to grayscale for SSIM
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2

    # Calculate metrics
    metrics = {}
    
    # PSNR (Peak Signal-to-Noise Ratio)
    metrics['PSNR'] = psnr(img1, img2)
    
    # SSIM (Structural Similarity Index)
    metrics['SSIM'] = ssim(gray1, gray2)
    
    return metrics

def main():
    # Read images
    clean = cv2.imread('sample.jpg')
    noisy = cv2.imread('noisy.png')
    denoised = cv2.imread('denoised.png')
    
    if clean is None or noisy is None or denoised is None:
        print("Error: Could not read one or more images")
        return
    
    # Resize clean image to match noisy/denoised dimensions if needed
    if clean.shape != noisy.shape:
        clean = cv2.resize(clean, (noisy.shape[1], noisy.shape[0]))
    
    # Calculate metrics for noisy image vs clean
    noisy_metrics = calculate_metrics(noisy, clean)
    
    # Calculate metrics for denoised image vs clean
    denoised_metrics = calculate_metrics(denoised, clean)
    
    # Print results
    print("\nImage Quality Metrics (compared to clean reference):")
    print("-" * 60)
    print("Noisy Image:")
    print(f"PSNR: {noisy_metrics['PSNR']:.2f} dB")
    print(f"SSIM: {noisy_metrics['SSIM']:.4f}")
    
    print("\nDenoised Image:")
    print(f"PSNR: {denoised_metrics['PSNR']:.2f} dB")
    print(f"SSIM: {denoised_metrics['SSIM']:.4f}")
    
    # Calculate improvement
    psnr_improvement = denoised_metrics['PSNR'] - noisy_metrics['PSNR']
    ssim_improvement = denoised_metrics['SSIM'] - noisy_metrics['SSIM']
    
    print("\nImprovement after denoising:")
    print("-" * 60)
    print(f"PSNR Improvement: {psnr_improvement:+.2f} dB")
    print(f"SSIM Improvement: {ssim_improvement:+.4f}")

if __name__ == '__main__':
    main() 