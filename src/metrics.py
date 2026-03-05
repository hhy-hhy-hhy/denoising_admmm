import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict, Optional
import numpy as np

class Metrics:
    """图像质量评估指标"""
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, restored: np.ndarray) -> float:
        """计算峰值信噪比"""
        # 确保图像在[0, 1]范围内
        original = np.clip(original, 0, 1)
        restored = np.clip(restored, 0, 1)
        return peak_signal_noise_ratio(original, restored, data_range=1.0)
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, restored: np.ndarray) -> float:
        """计算结构相似性"""
        original = np.clip(original, 0, 1)
        restored = np.clip(restored, 0, 1)
        return structural_similarity(original, restored, data_range=1.0)
    
    @staticmethod
    def calculate_all_metrics(original: np.ndarray, 
                             restored: np.ndarray,
                             mask: np.ndarray = None) -> Dict:
        """计算所有评估指标"""
        metrics = {}
        
        # 计算全局指标
        metrics['psnr'] = Metrics.calculate_psnr(original, restored)
        metrics['ssim'] = Metrics.calculate_ssim(original, restored)
        
        # 如果提供掩膜，计算缺失区域的指标
        if mask is not None:
            missing_mask = (mask == 0).astype(bool)
            if np.any(missing_mask):
                original_missing = original[missing_mask]
                restored_missing = restored[missing_mask]
                
                metrics['psnr_missing'] = peak_signal_noise_ratio(
                    original_missing, restored_missing, data_range=1.0
                )
                
                # 对于SSIM，需要提取补丁，这里简化处理
                metrics['mse_missing'] = np.mean((original_missing - restored_missing) ** 2)
        
        return metrics
