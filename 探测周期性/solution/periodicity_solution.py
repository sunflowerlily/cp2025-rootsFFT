#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 简化版

本模块实现太阳黑子数据的加载、可视化和周期性分析。
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
    data = np.loadtxt(filename, usecols=(2, 3))
    return data[:, 0], data[:, 1]  # 返回小数年份和黑子数

def plot_time_series(times, sunspots):
    """
    绘制太阳黑子数量随时间变化
    
    参数:
        times (numpy.ndarray): 时间数组
        sunspots (numpy.ndarray): 黑子数数组
    
    返回:
        matplotlib.figure.Figure: 图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, sunspots, 'b-')
    ax.set_title('Sunspot Number Variation')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sunspot Count')
    ax.grid(True)
    return fig

def compute_fft(sunspots):
    """
    计算太阳黑子数据的傅里叶变换
    
    参数:
        sunspots (numpy.ndarray): 黑子数数组
    
    返回:
        tuple: (频率数组, 功率谱数组)
    """
    N = len(sunspots)
    fft_values = np.fft.rfft(sunspots)
    power = np.abs(fft_values)**2
    freqs = np.fft.rfftfreq(N, d=1)
    
    # 调整功率谱(排除直流分量)
    if N > 1:
        power[1:] *= 2 if N % 2 == 0 else 2
    
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(frequencies, power, 'r-')
    ax.set_title('Power Spectrum')
    ax.set_xlabel('Frequency (1/month)')
    ax.set_ylabel('Power')
    ax.grid(True)
    
    # 检测主周期(排除直流分量)
    if len(frequencies) > 1:
        # 限制搜索范围在0.005-0.1 1/month (对应周期10-200个月)
        valid_range = (frequencies > 0.005) & (frequencies < 0.1)
        search_freqs = frequencies[valid_range]
        search_power = power[valid_range]
        
        if len(search_freqs) > 0:
            # 使用argmax找到功率谱中的最大值位置
            main_idx = np.argmax(search_power)
            main_freq = search_freqs[main_idx]
            main_period = 1/main_freq
            
            # 在图上标记主周期
            ax.plot(main_freq, search_power[main_idx], 'go')
            ax.text(main_freq, search_power[main_idx]*1.1, 
                   f'Main Period: {main_period:.1f} months',
                   ha='center')
            return fig, main_period
    
    return fig, 0

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
    print(f"Detected main period: {period:.1f} months ({period/12:.2f} years)")

if __name__ == "__main__":
    main()