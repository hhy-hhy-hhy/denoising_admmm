import numpy as np
import cv2
from typing import Tuple, Dict
import time

class BM3DInpainting:
    """BM3D图像修复算法"""
    
    def __init__(self, sigma: float = 25.0,
                 max_iter: int = 5,
                 verbose: bool = True):
        """
        参数:
        - sigma: 噪声标准差估计
        - max_iter: BM3D迭代次数
        """
        self.sigma = sigma
        self.max_iter = max_iter
        self.verbose = verbose
    
    def bm3d_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        使用BM3D进行去噪
        注意：需要安装BM3D库，可以使用以下命令：
        pip install bm3d
        
        如果没有安装，可以使用OpenCV的快速非局部均值去噪作为替代
        """
        try:
            import bm3d
            # 将图像缩放到[0, 255]范围
            image_scaled = (image * 255).astype(np.uint8)
            denoised = bm3d.bm3d(image_scaled, self.sigma)
            return denoised.astype(np.float32) / 255.0
        except ImportError:
            if self.verbose:
                print("BM3D library not found. Using OpenCV's fastNlMeansDenoising as fallback.")
            
            # 使用OpenCV的快速非局部均值去噪
            image_scaled = (image * 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(image_scaled, None, 
                                               h=self.sigma, 
                                               templateWindowSize=7,
                                               searchWindowSize=21)
            return denoised.astype(np.float32) / 255.0
    
    def solve(self, corrupted: np.ndarray, mask: np.ndarray,
              verbose: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        BM3D图像修复算法
        
        策略：
        1. 用均值初始化缺失像素
        2. 迭代应用BM3D去噪
        3. 在每次迭代后，将已知像素重置为原始值
        """
        start_time = time.time()
        
        # 初始化：用图像均值填充缺失像素
        image_mean = np.mean(corrupted[mask == 1]) if np.any(mask == 1) else 0.5
        x_init = corrupted.copy()
        x_init[mask == 0] = image_mean
        
        x = x_init.copy()
        
        if verbose:
            print(f"Starting BM3D inpainting with {self.max_iter} iterations...")
        
        for i in range(self.max_iter):
            if verbose:
                print(f"Iteration {i+1}/{self.max_iter}")
            
            # 使用BM3D去噪
            x_denoised = self.bm3d_denoise(x)
            
            # 保持已知像素不变
            x = mask * corrupted + (1 - mask) * x_denoised
        
        total_time = time.time() - start_time
        
        info = {
            'iterations': self.max_iter,
            'total_time': total_time,
            'final_psnr': None
        }
        
        return x, info
    
    def set_original(self, original: np.ndarray):
        """设置原始图像（用于计算PSNR）"""
        self.original = original
