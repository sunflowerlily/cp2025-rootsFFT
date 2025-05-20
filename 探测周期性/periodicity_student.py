#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探测周期性 - 太阳黑子数据分析

本模块实现了对太阳黑子数据的周期性分析。
"""

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import os


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
    # TODO: 实现数据加载代码 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用np.loadtxt加载数据，注意选择正确的列
    # 月平均数据格式: 年份 月份 小数年份 月平均黑子数 标准差 观测数 观测站数
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
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
    # TODO: 实现数据可视化代码 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用plt.plot绘制时间序列，添加适当的标签和标题
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return fig


def compute_fft(sunspots):
    """
    计算太阳黑子数据的傅里叶变换
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数量数组
    
    返回:
        tuple: (频率数组, 功率谱数组)
    """
    # TODO: 实现傅里叶变换计算代码 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用np.fft.fft计算傅里叶变换，计算功率谱和对应的频率
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
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
    # TODO: 实现功率谱可视化和峰值检测代码 (约20行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用plt.semilogy绘制功率谱，找出最显著峰值并计算对应周期
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
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


if __name__ == "__main__":
    main()