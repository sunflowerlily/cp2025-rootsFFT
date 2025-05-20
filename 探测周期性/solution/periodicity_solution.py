#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探测周期性 - 太阳黑子数据分析（参考答案）

本模块实现了对太阳黑子数据的周期性分析。
"""

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import os
from scipy.signal import find_peaks


def download_sunspot_data(url=None, local_filename='sunspot_data.txt'):
    """
    下载太阳黑子数据，如果本地已有文件则不重新下载
    
    参数:
        url (str): 数据下载地址，默认为None，将使用预设地址
        local_filename (str): 本地保存的文件名
    
    返回:
        str: 本地文件路径
    """
    # 默认使用月平均数据的URL
    if url is None:
        url = "http://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt"
    
    # 检查本地文件是否已存在
    if os.path.exists(local_filename):
        print(f"使用本地文件: {local_filename}")
        return local_filename
    
    # 下载数据
    print(f"从 {url} 下载数据...")
    with urlopen(url) as response, open(local_filename, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(f"数据已保存到: {local_filename}")
    
    return local_filename


def load_sunspot_data(filename):
    """
    加载太阳黑子数据文件
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        tuple: (时间数组, 太阳黑子数量数组)
    """
    # 加载数据，月平均数据格式: 年份 月份 小数年份 月平均黑子数 标准差 观测数 观测站数
    data = np.loadtxt(filename)
    
    # 提取年份、月份和黑子数
    years = data[:, 0]
    months = data[:, 1]
    sunspots = data[:, 3]  # 第4列是月平均黑子数
    
    # 创建时间数组（小数年份）
    times = years + (months - 1) / 12
    
    return times, sunspots


def plot_sunspot_data(times, sunspots):
    """
    绘制太阳黑子数量随时间的变化图
    
    参数:
        times (numpy.ndarray): 时间数组（年份）
        sunspots (numpy.ndarray): 太阳黑子数量数组
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制时间序列
    ax.plot(times, sunspots, 'b-', linewidth=1)
    
    # 添加标题和标签
    ax.set_title('太阳黑子数量随时间的变化 (1749-至今)')
    ax.set_xlabel('年份')
    ax.set_ylabel('月平均黑子数')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 估计周期
    # 通过观察图像中的峰值间隔来估计周期
    # 这里我们可以标记一些明显的峰值
    peak_years = [1778, 1789, 1804, 1816, 1830, 1837, 1848, 1860, 1870, 1883, 1893, 1905, 1917, 1928, 1937, 1947, 1957, 1968, 1979, 1989, 2000, 2014]
    peak_indices = [np.abs(times - year).argmin() for year in peak_years]
    
    # 计算相邻峰值之间的平均间隔
    if len(peak_years) > 1:
        avg_period_years = np.mean(np.diff(peak_years))
        avg_period_months = avg_period_years * 12
        
        # 在图上标记估计的周期
        ax.text(0.02, 0.95, f'估计周期: {avg_period_years:.2f} 年 ({avg_period_months:.1f} 个月)',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig


def compute_fft(sunspots):
    """
    计算太阳黑子数据的傅里叶变换
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数量数组
    
    返回:
        tuple: (频率数组, 功率谱数组)
    """
    # 数据长度
    N = len(sunspots)
    
    # 计算傅里叶变换
    fft_values = np.fft.fft(sunspots)
    
    # 计算功率谱 (平方模)
    power_spectrum = np.abs(fft_values)**2
    
    # 创建频率数组
    # 对于长度为N的信号，频率范围是[0, 1, 2, ..., N/2, ..., N-1]
    # 其中N/2+1到N-1对应负频率
    frequencies = np.fft.fftfreq(N)
    
    # 我们只关注正频率部分（包括零频率）
    positive_freq_idx = np.arange(1, N // 2 + 1)  # 排除零频率
    frequencies = frequencies[positive_freq_idx]
    power_spectrum = power_spectrum[positive_freq_idx]
    
    return frequencies, power_spectrum


def plot_power_spectrum(frequencies, power_spectrum):
    """
    绘制功率谱
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power_spectrum (numpy.ndarray): 功率谱数组
    
    返回:
        tuple: (matplotlib.figure.Figure, 主要峰值对应的周期)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制功率谱（使用对数刻度）
    ax.semilogy(frequencies, power_spectrum, 'r-')
    
    # 添加标题和标签
    ax.set_title('太阳黑子数据的功率谱')
    ax.set_xlabel('频率 (周期^-1)')
    ax.set_ylabel('功率谱 |c_k|^2')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 找出最显著的峰值（排除零频率附近）
    # 我们只考虑频率大于0.01的部分（对应周期小于数据长度的100倍）
    min_freq_idx = np.where(frequencies > 0.01)[0][0]
    peaks, _ = find_peaks(power_spectrum[min_freq_idx:], height=np.max(power_spectrum[min_freq_idx:]) * 0.5)
    peaks = peaks + min_freq_idx  # 调整索引
    
    # 按功率排序峰值
    sorted_peaks = sorted(peaks, key=lambda i: power_spectrum[i], reverse=True)
    
    # 获取最显著的峰值
    if sorted_peaks:
        main_peak_idx = sorted_peaks[0]
        main_peak_freq = frequencies[main_peak_idx]
        main_period = 1 / main_peak_freq  # 周期 = 1/频率
        
        # 在图上标记主要峰值
        ax.plot(main_peak_freq, power_spectrum[main_peak_idx], 'go', markersize=10)
        ax.text(main_peak_freq, power_spectrum[main_peak_idx] * 1.1,
                f'主要周期: {main_period:.1f} 个月',
                fontsize=12, ha='center')
    else:
        main_period = 0  # 未找到显著峰值
    
    # 添加次要峰值（如果有）
    for i, peak_idx in enumerate(sorted_peaks[1:3]):  # 最多显示接下来的2个峰值
        if i < 2:  # 限制为2个次要峰值
            peak_freq = frequencies[peak_idx]
            period = 1 / peak_freq
            ax.plot(peak_freq, power_spectrum[peak_idx], 'bo', markersize=8)
            ax.text(peak_freq, power_spectrum[peak_idx] * 1.1,
                    f'周期: {period:.1f} 个月',
                    fontsize=10, ha='center')
    
    return fig, main_period


def main():
    """
    主函数，执行太阳黑子数据的周期性分析
    """
    # 1. 下载并加载数据
    data_file = download_sunspot_data()
    times, sunspots = load_sunspot_data(data_file)
    
    # 2. 绘制原始数据
    fig_time = plot_sunspot_data(times, sunspots)
    plt.savefig('sunspot_time_series.png', dpi=300)
    plt.show()
    
    # 3. 计算傅里叶变换
    frequencies, power_spectrum = compute_fft(sunspots)
    
    # 4. 绘制功率谱并找出主要周期
    fig_spectrum, main_period = plot_power_spectrum(frequencies, power_spectrum)
    plt.savefig('sunspot_power_spectrum.png', dpi=300)
    plt.show()
    
    # 5. 输出结果
    print(f"\n太阳黑子的主要周期: {main_period:.2f} 个月")
    print(f"太阳黑子的主要周期: {main_period/12:.2f} 年")
    
    # 6. 比较目视估计和傅里叶分析结果
    # 计算目视估计的周期（基于峰值年份）
    peak_years = [1778, 1789, 1804, 1816, 1830, 1837, 1848, 1860, 1870, 1883, 1893, 1905, 1917, 1928, 1937, 1947, 1957, 1968, 1979, 1989, 2000, 2014]
    if len(peak_years) > 1:
        avg_period_years = np.mean(np.diff(peak_years))
        avg_period_months = avg_period_years * 12
        print(f"\n目视估计的周期: {avg_period_years:.2f} 年 ({avg_period_months:.1f} 个月)")
        print(f"傅里叶分析的周期: {main_period/12:.2f} 年 ({main_period:.1f} 个月)")
        print(f"相对误差: {abs(avg_period_months - main_period) / avg_period_months * 100:.2f}%")


if __name__ == "__main__":
    main()