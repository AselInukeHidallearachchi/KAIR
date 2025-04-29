import cv2
import numpy as np
import argparse

def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to an image."""
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./clean.png', help='Path to clean image')
    parser.add_argument('--output', type=str, default='./noisy.png', help='Path to save noisy image')
    parser.add_argument('--sigma', type=float, default=15, help='Noise level (sigma)')
    args = parser.parse_args()
    
    # Load clean image
    img = cv2.imread(args.input)
    if img is None:
        print(f'Could not read image: {args.input}')
        return
    
    # Add noise
    noisy_img = add_gaussian_noise(img, args.sigma)
    
    # Save noisy image
    cv2.imwrite(args.output, noisy_img)
    print(f'Noisy image saved to {args.output}')

if __name__ == '__main__':
    main()