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

@dataclass
class Line:
    """线段障碍物，由两个端点p1和p2定义"""
    p1: np.ndarray
    p2: np.ndarray