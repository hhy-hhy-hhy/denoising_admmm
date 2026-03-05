import numpy as np
from typing import Tuple, Dict, List
import time
from tqdm import tqdm

class FISTAL1:
    """基于L1正则化的FISTA图像修复算法"""
    
    def __init__(self, lambda_: float = 0.1, 
                 max_iter: int = 500, 
                 tol: float = 1e-4,
                 step_size: float = 1.0):
        """
        参数:
        - lambda_: L1正则化参数
        - max_iter: 最大迭代次数
        - tol: 收敛容差
        - step_size: 步长
        """
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        
    def soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """软阈值函数"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def wavelet_transform(self, image: np.ndarray, level: int = 3):
        """小波变换（使用PyWavelets）"""
        import pywt
        coeffs = pywt.wavedec2(image, 'db4', level=level)
        return coeffs
    
    def inverse_wavelet_transform(self, coeffs):
        """逆小波变换"""
        import pywt
        return pywt.waverec2(coeffs, 'db4')
    
    def objective_function(self, x: np.ndarray, y: np.ndarray, 
                          mask: np.ndarray) -> float:
        """目标函数：数据保真项 + L1正则项"""
        # 数据保真项（仅在已知像素上）
        data_fidelity = 0.5 * np.sum(mask * (x - y) ** 2)
        
        # L1正则项（在小波域）
        coeffs = self.wavelet_transform(x)
        l1_penalty = self.lambda_ * np.sum([np.sum(np.abs(c)) for c in coeffs])
        
        return data_fidelity + l1_penalty
    
    def solve(self, corrupted: np.ndarray, mask: np.ndarray, 
              verbose: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        使用FISTA算法进行图像修复
        
        参数:
        - corrupted: 损坏的图像
        - mask: 掩膜（1表示已知，0表示缺失）
        - verbose: 是否显示进度
        
        返回:
        - restored: 修复后的图像
        - info: 包含收敛信息的字典
        """
        # 初始化
        x = corrupted.copy()
        x_prev = x.copy()
        t = 1.0
        t_prev = 1.0
        
        # 存储收敛信息
        objective_values = []
        psnr_values = []
        timestamps = []
        
        start_time = time.time()
        
        # 创建迭代器
        iterator = range(self.max_iter)
        if verbose:
            iterator = tqdm(iterator, desc="FISTA-L1 Iterations")
        
        for iteration in iterator:
            # FISTA加速步
            t = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
            z = x + ((t_prev - 1) / t) * (x - x_prev)
            
            # 梯度步（仅在已知像素上更新）
            gradient = mask * (z - corrupted)
            
            # 更新变量
            x_prev = x.copy()
            x_temp = z - self.step_size * gradient
            
            # 小波变换
            coeffs = self.wavelet_transform(x_temp)
            
            # 在小波域应用软阈值
            threshold = self.step_size * self.lambda_
            coeffs_thresh = []
            for coeff in coeffs:
                if isinstance(coeff, tuple):  # 细节系数
                    coeffs_thresh.append(tuple(
                        self.soft_threshold(c, threshold) for c in coeff
                    ))
                else:  # 近似系数
                    coeffs_thresh.append(coeff)
            
            # 逆小波变换
            x = self.inverse_wavelet_transform(coeffs_thresh)
            
            # 确保在已知像素处保持原值
            x = mask * corrupted + (1 - mask) * x
            
            # 计算目标函数值
            obj_val = self.objective_function(x, corrupted, mask)
            objective_values.append(obj_val)
            
            # 计算当前PSNR（相对于原始图像，如果有的话）
            if hasattr(self, 'original'):
                from skimage.metrics import peak_signal_noise_ratio
                psnr = peak_signal_noise_ratio(self.original, x)
                psnr_values.append(psnr)
            
            # 记录时间
            timestamps.append(time.time() - start_time)
            
            # 检查收敛
            if iteration > 0:
                relative_change = np.abs(objective_values[-1] - objective_values[-2]) / \
                                 (np.abs(objective_values[-2]) + 1e-10)
                if relative_change < self.tol:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            t_prev = t
        
        # 准备返回信息
        total_time = time.time() - start_time
        
        info = {
            'objective_values': np.array(objective_values),
            'psnr_values': np.array(psnr_values) if psnr_values else None,
            'iterations': iteration + 1,
            'total_time': total_time,
            'converged': iteration < self.max_iter - 1,
            'final_objective': objective_values[-1],
            'timestamps': np.array(timestamps)
        }
        
        return x, info
    
    def set_original(self, original: np.ndarray):
        """设置原始图像（用于计算PSNR）"""
        self.original = original
