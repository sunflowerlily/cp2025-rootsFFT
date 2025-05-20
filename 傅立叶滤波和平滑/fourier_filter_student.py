#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析

本模块实现了对道Jones工业平均指数数据的傅立叶分析和滤波处理。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载道Jones工业平均指数数据
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        numpy.ndarray: 指数数组
    """
    # TODO: 实现数据加载功能 (约5行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用np.loadtxt加载数据文件，处理可能的异常
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return data

def plot_data(data, title="Dow Jones Industrial Average"):
    """
    绘制时间序列数据
    
    参数:
        data (numpy.ndarray): 输入数据数组
        title (str): 图表标题
    
    返回:
        None
    """
    # TODO: 实现数据可视化 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用plt.plot绘制数据，添加适当的标签和标题
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))

def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波
    
    参数:
        data (numpy.ndarray): 输入数据数组
        keep_fraction (float): 保留的傅立叶系数比例
    
    返回:
        tuple: (滤波后的数据数组, 原始傅立叶系数数组)
    """
    # TODO: 实现傅立叶滤波功能 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 
    # 1. 使用np.fft.rfft计算实数傅立叶变换
    # 2. 根据keep_fraction计算保留的系数数量
    # 3. 创建滤波后的系数数组
    # 4. 使用np.fft.irfft计算逆变换
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return filtered_data, fft_coeff

def plot_comparison(original, filtered, title="Fourier Filter Result"):
    """
    绘制原始数据和滤波结果的比较
    
    参数:
        original (numpy.ndarray): 原始数据数组
        filtered (numpy.ndarray): 滤波后的数据数组
        title (str): 图表标题
    
    返回:
        None
    """
    # TODO: 实现数据比较可视化 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 
    # 1. 使用不同颜色绘制原始和滤波数据
    # 2. 添加图例、标签和标题
    # 3. 使用plt.grid添加网格线
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))

def main():
    # 任务1：数据加载与可视化
    data = load_data('dow.txt')
    plot_data(data, "Dow Jones Industrial Average - Original Data")
    
    # 任务2：傅立叶变换与滤波（保留前10%系数）
    filtered_10, coeff = fourier_filter(data, 0.1)
    plot_comparison(data, filtered_10, "Fourier Filter (Keep Top 10% Coefficients)")
    
    # 任务3：修改滤波参数（保留前2%系数）
    filtered_2, _ = fourier_filter(data, 0.02)
    plot_comparison(data, filtered_2, "Fourier Filter (Keep Top 2% Coefficients)")

if __name__ == "__main__":
    main()