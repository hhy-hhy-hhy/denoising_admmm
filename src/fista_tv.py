import numpy as np
from typing import Tuple, Dict
import time
from tqdm import tqdm

class FISTATV:
    """基于全变分(TV)正则化的FISTA图像修复算法"""
    
    def __init__(self, lambda_: float = 0.1, 
                 max_iter: int = 500,
                 tol: float = 1e-4,
                 step_size: float = 1.0):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
    
    def gradient_operator(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算图像梯度"""
        grad_x = np.roll(image, -1, axis=1) - image
        grad_y = np.roll(image, -1, axis=0) - image
        return grad_x, grad_y
    
    def divergence_operator(self, grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
        """计算散度（梯度的负共轭）"""
        div_x = grad_x - np.roll(grad_x, 1, axis=1)
        div_y = grad_y - np.roll(grad_y, 1, axis=0)
        return div_x + div_y
    
    def tv_norm(self, image: np.ndarray) -> float:
        """计算总变分范数"""
        grad_x, grad_y = self.gradient_operator(image)
        tv = np.sum(np.sqrt(grad_x**2 + grad_y**2 + 1e-8))
        return tv
    
    def objective_function(self, x: np.ndarray, y: np.ndarray, 
                          mask: np.ndarray) -> float:
        """目标函数：数据保真项 + TV正则项"""
        # 数据保真项
        data_fidelity = 0.5 * np.sum(mask * (x - y) ** 2)
        
        # TV正则项
        tv_penalty = self.lambda_ * self.tv_norm(x)
        
        return data_fidelity + tv_penalty
    
    def solve(self, corrupted: np.ndarray, mask: np.ndarray,
              verbose: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        使用FISTA-TV算法进行图像修复
        """
        # 初始化
        x = corrupted.copy()
        p_x = np.zeros_like(x)
        p_y = np.zeros_like(x)
        
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
            iterator = tqdm(iterator, desc="FISTA-TV Iterations")
        
        for iteration in iterator:
            # FISTA加速步
            t = (1 + np.sqrt(1 + 4 * t_prev ** 2)) / 2
            z = x + ((t_prev - 1) / t) * (x - x_prev)
            
            # 对偶变量更新
            grad_x, grad_y = self.gradient_operator(z)
            p_x_new = p_x + self.step_size * grad_x
            p_y_new = p_y + self.step_size * grad_y
            
            # 投影到单位圆
            norm = np.sqrt(p_x_new**2 + p_y_new**2 + 1e-8)
            p_x_new = p_x_new / np.maximum(norm / self.lambda_, 1)
            p_y_new = p_y_new / np.maximum(norm / self.lambda_, 1)
            
            # 原始变量更新
            div = self.divergence_operator(p_x_new, p_y_new)
            x_new = z - self.step_size * div
            
            # 确保在已知像素处保持原值
            x_new = mask * corrupted + (1 - mask) * x_new
            
            # 更新变量
            x_prev = x.copy()
            x = x_new.copy()
            p_x = p_x_new.copy()
            p_y = p_y_new.copy()
            
            # 计算目标函数值
            obj_val = self.objective_function(x, corrupted, mask)
            objective_values.append(obj_val)
            
            # 计算PSNR
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
        """设置原始图像"""
        self.original = original
