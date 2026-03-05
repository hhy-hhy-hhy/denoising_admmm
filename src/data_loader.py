
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

class Set14Dataset:
    """Set14数据集加载器"""
    
    def __init__(self, data_dir: str = "data/Set14"):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob("*.png"))
        self.images = {}
        
    def load_images(self) -> Dict[str, np.ndarray]:
        """加载所有图像"""
        for img_path in self.image_paths:
            img_name = img_path.stem
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 归一化到[0, 1]
                img = img.astype(np.float32) / 255.0
                self.images[img_name] = img
        return self.images
    
    def create_masks(self, image_shape: Tuple[int, int], 
                     missing_ratios: List[float] = [0.25, 0.5, 0.75],
                     seed: int = 42) -> Dict[float, np.ndarray]:
        """创建随机缺失掩膜"""
        np.random.seed(seed)
        masks = {}
        
        for ratio in missing_ratios:
            mask = np.ones(image_shape, dtype=np.float32)
            num_pixels = int(np.prod(image_shape) * ratio)
            
            # 随机选择像素位置
            flat_indices = np.random.choice(np.prod(image_shape), 
                                           num_pixels, 
                                           replace=False)
            rows, cols = np.unravel_index(flat_indices, image_shape)
            mask[rows, cols] = 0  # 0表示缺失
            
            masks[ratio] = mask
        
        return masks
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """应用掩膜到图像"""
        corrupted = image.copy()
        corrupted[mask == 0] = 0  # 设置缺失像素为0
        return corrupted
    
    def visualize_corruption(self, image_name: str, 
                            missing_ratios: List[float] = [0.25, 0.5, 0.75]):
        """可视化不同缺失比例的效果"""
        if image_name not in self.images:
            raise ValueError(f"Image {image_name} not found")
        
        image = self.images[image_name]
        masks = self.create_masks(image.shape, missing_ratios)
        
        fig, axes = plt.subplots(1, len(missing_ratios) + 1, 
                                figsize=(15, 4))
        
        # 显示原始图像
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Original: {image_name}')
        axes[0].axis('off')
        
        # 显示不同缺失比例的图像
        for idx, (ratio, mask) in enumerate(masks.items(), 1):
            corrupted = self.apply_mask(image, mask)
            axes[idx].imshow(corrupted, cmap='gray')
            axes[idx].set_title(f'Missing: {ratio*100:.0f}%')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/visual/corruption_{image_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
