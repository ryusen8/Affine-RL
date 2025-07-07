import numpy as np
from dataclasses import dataclass

@dataclass
class Circle:
    """圆形障碍物"""
    center: np.ndarray  # [x, y]
    radius: float

@dataclass
class Rectangle:
    """矩形障碍物"""
    center: np.ndarray  # [x, y]
    size: np.ndarray    # [width, height]
    angle: float        # 旋转角度 (弧度)