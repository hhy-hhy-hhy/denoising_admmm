
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, metrics
from skimage.util import random_noise
import pywt
from bm3d import bm3d  # If installation fails, replace with: from skimage.restoration import denoise_bm3d as bm3d

# -------------------- Utility Functions --------------------
def load_and_noise(image_path, sigma=25):
    """Load grayscale image and add Gaussian noise."""
    img = io.imread(image_path, as_gray=True).astype(np.float64)
    noisy = random_noise(img, mode='gaussian', var=(sigma/255.)**2)
    return img, noisy

def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio."""
    return metrics.peak_signal_noise_ratio(img1, img2, data_range=1.0)

def ssim(img1, img2):
    """Structural Similarity Index."""
    return metrics.structural_similarity(img1, img2, data_range=1.0)

def estimate_noise_sigma(noisy, wavelet='db4'):
    """Estimate noise standard deviation using median of finest wavelet details."""
    coeffs = pywt.wavedec2(noisy, wavelet, level=None)
    # Take the finest level (most detailed) coefficients in three orientations
    cH, cV, cD = coeffs[-1]
    detail = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
    sigma = np.median(np.abs(detail)) / 0.6745
    return sigma

# -------------------- ADMM-TV Denoising --------------------
def admm_tv_denoise(noisy, lambd=0.05, rho=0.1, max_iter=50):
    """ADMM for Total Variation denoising (isotropic TV) with FFT solver."""
    x = noisy.copy()
    # y and u are tuples of two components (x-gradient, y-gradient)
    y = (np.zeros_like(noisy), np.zeros_like(noisy))
    u = (np.zeros_like(noisy), np.zeros_like(noisy))

    def Dx(x): return np.roll(x, -1, axis=1) - x
    def Dy(x): return np.roll(x, -1, axis=0) - x
    def Dxt(y): return np.roll(y, 1, axis=1) - y
    def Dyt(y): return np.roll(y, 1, axis=0) - y

    def shrink(z, threshold):
        norm = np.sqrt(z[0]**2 + z[1]**2)
        factor = np.maximum(1 - threshold / (norm + 1e-10), 0)
        return (z[0] * factor, z[1] * factor)

    from numpy.fft import fft2, ifft2
    h, w = noisy.shape
    wx = np.fft.fftfreq(w).reshape(1, -1)
    wy = np.fft.fftfreq(h).reshape(-1, 1)
    Kx = 1 - np.exp(-2j * np.pi * wx)
    Ky = 1 - np.exp(-2j * np.pi * wy)
    H = 1 + rho * (np.abs(Kx)**2 + np.abs(Ky)**2)   # fixed denominator

    for i in range(max_iter):
        yx, yy = y
        ux, uy = u
        rhs = noisy + rho * (Dxt(yx - ux) + Dyt(yy - uy))
        x = np.real(ifft2(fft2(rhs) / H))

        Dxx = Dx(x)
        Dyy = Dy(x)
        vx = Dxx + ux
        vy = Dyy + uy
        sx, sy = shrink((vx, vy), lambd / rho)
        y = (sx, sy)

        ux = ux + Dxx - sx
        uy = uy + Dyy - sy
        u = (ux, uy)
    return x

# -------------------- ISTA / FISTA Denoising (Wavelet) --------------------
def ista_wavelet_denoise(noisy, wavelet='db4', lambd=None, step_size=1, max_iter=100, return_history=False):
    """
    ISTA for wavelet-based denoising.
    If lambd is None, use universal threshold from estimated noise sigma.
    """
    if lambd is None:
        sigma_hat = estimate_noise_sigma(noisy, wavelet)
        lambd = sigma_hat * np.sqrt(2 * np.log(noisy.size))
    x = noisy.copy()
    psnr_history = [] if return_history else None
    for _ in range(max_iter):
        grad = x - noisy
        v = x - step_size * grad
        # Wavelet soft thresholding
        coeffs = pywt.wavedec2(v, wavelet, level=None)
        coeffs_thresh = list(coeffs)
        coeffs_thresh[0] = coeffs[0]   # keep approximation
        for j in range(1, len(coeffs)):
            cH, cV, cD = coeffs[j]
            cH = pywt.threshold(cH, step_size * lambd, mode='soft')
            cV = pywt.threshold(cV, step_size * lambd, mode='soft')
            cD = pywt.threshold(cD, step_size * lambd, mode='soft')
            coeffs_thresh[j] = (cH, cV, cD)
        x_new = pywt.waverec2(coeffs_thresh, wavelet)
        x_new = x_new[:noisy.shape[0], :noisy.shape[1]]
        x = x_new
        if return_history:
            # To record PSNR we would need the clean image; we skip here.
            # History is handled externally in the demonstration below.
            pass
    return x

def fista_wavelet_denoise(noisy, wavelet='db4', lambd=None, step_size=1, max_iter=100):
    """FISTA for wavelet-based denoising."""
    if lambd is None:
        sigma_hat = estimate_noise_sigma(noisy, wavelet)
        lambd = sigma_hat * np.sqrt(2 * np.log(noisy.size))
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    for k in range(max_iter):
        x_old = x.copy()
        grad = y - noisy
        v = y - step_size * grad
        coeffs = pywt.wavedec2(v, wavelet, level=None)
        coeffs_thresh = list(coeffs)
        coeffs_thresh[0] = coeffs[0]
        for j in range(1, len(coeffs)):
            cH, cV, cD = coeffs[j]
            cH = pywt.threshold(cH, step_size * lambd, mode='soft')
            cV = pywt.threshold(cV, step_size * lambd, mode='soft')
            cD = pywt.threshold(cD, step_size * lambd, mode='soft')
            coeffs_thresh[j] = (cH, cV, cD)
        x = pywt.waverec2(coeffs_thresh, wavelet)
        x = x[:noisy.shape[0], :noisy.shape[1]]
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
    return x

# -------------------- BM3D Denoising --------------------
def bm3d_denoise(noisy, sigma):
    """BM3D denoising (input image range [0,1], sigma in same range)."""
    return bm3d(noisy, sigma)

# -------------------- Main --------------------
if __name__ == '__main__':
    # Parameters
    sigma = 25
    image_path = 'face.png'   # Replace with your image path
    wavelet = 'db4'
    max_iter = 200
    step_size = 0.2

    # Load image
    img_clean, img_noisy = load_and_noise(image_path, sigma)
    print(f'Noisy PSNR: {psnr(img_clean, img_noisy):.2f} dB')

    # Run denoising algorithms
    print('Running ADMM-TV...')
    img_admm = admm_tv_denoise(img_noisy, lambd=0.05, rho=0.1, max_iter=50)

    print('Running ISTA...')
    img_ista = ista_wavelet_denoise(img_noisy, wavelet=wavelet, lambd=None,
                                    step_size=step_size, max_iter=max_iter)

    print('Running FISTA...')
    img_fista = fista_wavelet_denoise(img_noisy, wavelet=wavelet, lambd=None,
                                      step_size=step_size, max_iter=max_iter)

    print('Running BM3D...')
    img_bm3d = bm3d_denoise(img_noisy, sigma/255.)

    # Compute metrics
    results = {
        'Noisy': (psnr(img_clean, img_noisy), ssim(img_clean, img_noisy)),
        'ADMM-TV': (psnr(img_clean, img_admm), ssim(img_clean, img_admm)),
        'ISTA': (psnr(img_clean, img_ista), ssim(img_clean, img_ista)),
        'FISTA': (psnr(img_clean, img_fista), ssim(img_clean, img_fista)),
        'BM3D': (psnr(img_clean, img_bm3d), ssim(img_clean, img_bm3d))
    }

    # Print table
    print('\nMethod\t\tPSNR (dB)\tSSIM')
    for method, (p, s) in results.items():
        print(f'{method}\t{p:.2f}\t\t{s:.4f}')

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.ravel()
    ax[0].imshow(img_clean, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].imshow(img_noisy, cmap='gray')
    ax[1].set_title(f'Noisy ({results["Noisy"][0]:.2f} dB)')
    ax[1].axis('off')

    ax[2].imshow(img_admm, cmap='gray')
    ax[2].set_title(f'ADMM-TV ({results["ADMM-TV"][0]:.2f} dB)')
    ax[2].axis('off')

    ax[3].imshow(img_ista, cmap='gray')
    ax[3].set_title(f'ISTA ({results["ISTA"][0]:.2f} dB)')
    ax[3].axis('off')

    ax[4].imshow(img_fista, cmap='gray')
    ax[4].set_title(f'FISTA ({results["FISTA"][0]:.2f} dB)')
    ax[4].axis('off')

    ax[5].imshow(img_bm3d, cmap='gray')
    ax[5].set_title(f'BM3D ({results["BM3D"][0]:.2f} dB)')
    ax[5].axis('off')

    plt.tight_layout()
    plt.savefig('denoising_comparison.png', dpi=300)
    plt.show()

    # Optional: Plot convergence curves for ISTA and FISTA (requires modifying functions to record PSNR history)
    # Below we demonstrate how to do that.

    def ista_with_history(noisy, clean, wavelet='db4', lambd=None, step_size=0.2, max_iter=100):
        """ISTA with PSNR history."""
        if lambd is None:
            sigma_hat = estimate_noise_sigma(noisy, wavelet)
            lambd = 0.8*sigma_hat * np.sqrt(2 * np.log(noisy.size))
        x = noisy.copy()
        history = []
        for _ in range(max_iter):
            grad = x - noisy
            v = x - step_size * grad
            coeffs = pywt.wavedec2(v, wavelet, level=None)
            coeffs_thresh = list(coeffs)
            coeffs_thresh[0] = coeffs[0]
            for j in range(1, len(coeffs)):
                cH, cV, cD = coeffs[j]
                cH = pywt.threshold(cH, step_size * lambd, mode='soft')
                cV = pywt.threshold(cV, step_size * lambd, mode='soft')
                cD = pywt.threshold(cD, step_size * lambd, mode='soft')
                coeffs_thresh[j] = (cH, cV, cD)
            x_new = pywt.waverec2(coeffs_thresh, wavelet)
            x_new = x_new[:noisy.shape[0], :noisy.shape[1]]
            x = x_new
            history.append(psnr(clean, x))
        return x, history

    def fista_with_history(noisy, clean, wavelet='db4', lambd=None, step_size=0.2, max_iter=100):
        """FISTA with PSNR history."""
        if lambd is None:
            sigma_hat = estimate_noise_sigma(noisy, wavelet)
            lambd = 0.8*sigma_hat * np.sqrt(2 * np.log(noisy.size))
        x = noisy.copy()
        y = x.copy()
        t = 1.0
        history = []
        for k in range(max_iter):
            x_old = x.copy()
            grad = y - noisy
            v = y - step_size * grad
            coeffs = pywt.wavedec2(v, wavelet, level=None)
            coeffs_thresh = list(coeffs)
            coeffs_thresh[0] = coeffs[0]
            for j in range(1, len(coeffs)):
                cH, cV, cD = coeffs[j]
                cH = pywt.threshold(cH, step_size * lambd, mode='soft')
                cV = pywt.threshold(cV, step_size * lambd, mode='soft')
                cD = pywt.threshold(cD, step_size * lambd, mode='soft')
                coeffs_thresh[j] = (cH, cV, cD)
            x = pywt.waverec2(coeffs_thresh, wavelet)
            x = x[:noisy.shape[0], :noisy.shape[1]]
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x + ((t - 1) / t_new) * (x - x_old)
            t = t_new
            history.append(psnr(clean, x))
        return x, history

    # Re-run with history
    print('\nGenerating convergence curves...')
    _, hist_ista = ista_with_history(img_noisy, img_clean, wavelet=wavelet, lambd=None,
                                     step_size=step_size, max_iter=max_iter)
    _, hist_fista = fista_with_history(img_noisy, img_clean, wavelet=wavelet, lambd=None,
                                       step_size=step_size, max_iter=max_iter)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_ista, label='ISTA')
    plt.plot(hist_fista, label='FISTA')
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_curve.png', dpi=300)
    plt.show()
