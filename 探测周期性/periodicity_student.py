#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 学生模板

本模块实现太阳黑子数据的加载、可视化和周期性分析。
学生需要完成所有标记为TODO的函数实现。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(filename):
    """
    加载太阳黑子数据文件
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        tuple: (时间数组, 太阳黑子数量数组)
    """
    # TODO: 实现数据加载功能 (约5行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用np.loadtxt加载数据，注意选择正确的列(第3列:小数年份, 第4列:黑子数)
    
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    
    return times, sunspots

def plot_time_series(times, sunspots):
    """
    绘制太阳黑子数量随时间变化
    
    参数:
        times (numpy.ndarray): 时间数组
        sunspots (numpy.ndarray): 黑子数数组
    
    返回:
        matplotlib.figure.Figure: 图形对象
    """
    # TODO: 实现时间序列可视化 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用plt.subplots创建图形，设置适当的标题和轴标签
    
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    
    return fig

def compute_fft(sunspots):
    """
    计算太阳黑子数据的傅里叶变换
    
    参数:
        sunspots (numpy.ndarray): 黑子数数组
    
    返回:
        tuple: (频率数组, 功率谱数组)
    """
    # TODO: 实现傅里叶变换计算 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用np.fft.rfft计算实数傅里叶变换，计算功率谱和对应频率
    
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    
    return freqs, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱并检测主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组(单位：1/月)
        power (numpy.ndarray): 功率谱数组
    
    返回:
        tuple: (图形对象, 主周期(月))
    """
    # TODO: 实现功率谱可视化和周期检测 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用plt.semilogy绘制对数功率谱，找出最显著峰值并计算周期
    
    raise NotImplementedError("请在 {} 中实现此函数".format(__file__))
    
    return fig, main_period

def main():
    """主分析流程"""
    # 1. 加载数据
    times, sunspots = load_sunspot_data('sunspot_data.txt')
    
    # 2. 绘制时间序列
    fig1 = plot_time_series(times, sunspots)
    plt.show()
    
    # 3. 计算并绘制功率谱
    freqs, power = compute_fft(sunspots)
    fig2, period = plot_power_spectrum(freqs, power)
    plt.show()
    
    # 4. 输出结果
    print(f"检测到的主要周期: {period:.1f} 个月 ({period/12:.2f} 年)")

if __name__ == "__main__":
    main()