#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析（参考答案）

本模块实现了对道琼斯工业平均指数数据的傅立叶分析和滤波处理。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime


def load_djia_data(filename):
    """
    加载道琼斯工业平均指数数据
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        tuple: (日期数组, 指数数组)
    """
    # 使用pandas读取CSV文件
    try:
        df = pd.read_csv(filename)
        
        # 检查必要的列是否存在
        required_columns = ['Date', 'Close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据文件缺少必要的列: {col}")
        
        # 转换日期列为datetime对象
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 按日期排序
        df = df.sort_values('Date')
        
        # 处理缺失值（如果有）
        if df['Close'].isna().any():
            print(f"警告: 数据中有 {df['Close'].isna().sum()} 个缺失值，使用前值填充")
            df['Close'] = df['Close'].fillna(method='ffill')
        
        # 提取日期和收盘价
        dates = df['Date'].values
        values = df['Close'].values
        
        return dates, values
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise


def plot_time_series(dates, values, title="道琼斯工业平均指数"):
    """
    绘制时间序列数据
    
    参数:
        dates (numpy.ndarray): 日期数组
        values (numpy.ndarray): 数值数组
        title (str): 图表标题
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制时间序列
    ax.plot(dates, values, 'b-', linewidth=1)
    
    # 添加标题和标签
    ax.set_title(title)
    ax.set_xlabel('日期')
    ax.set_ylabel('指数值')
    
    # 格式化x轴日期标签
    fig.autofmt_xdate()
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    fig.tight_layout()
    
    return fig


def compute_fft(values):
    """
    计算数据的傅立叶变换
    
    参数:
        values (numpy.ndarray): 输入数据数组
    
    返回:
        tuple: (频率数组, 傅立叶变换系数数组)
    """
    # 数据长度
    N = len(values)
    
    # 计算傅立叶变换
    fft_values = np.fft.fft(values)
    
    # 创建频率数组
    # 假设数据是每日数据，频率单位为每天的周期数
    frequencies = np.fft.fftfreq(N, d=1.0)
    
    return frequencies, fft_values


def plot_power_spectrum(frequencies, fft_values):
    """
    绘制功率谱
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        fft_values (numpy.ndarray): 傅立叶变换系数数组
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # 计算功率谱（平方模）
    power_spectrum = np.abs(fft_values)**2
    
    # 只关注正频率部分
    positive_freq_idx = frequencies > 0
    pos_frequencies = frequencies[positive_freq_idx]
    pos_power_spectrum = power_spectrum[positive_freq_idx]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 使用对数刻度绘制功率谱
    ax.semilogy(pos_frequencies, pos_power_spectrum, 'r-')
    
    # 添加标题和标签
    ax.set_title('道琼斯指数的功率谱')
    ax.set_xlabel('频率 (每天的周期数)')
    ax.set_ylabel('功率谱 |X(f)|²')
    
    # 添加第二个x轴，显示周期（天）
    ax2 = ax.twiny()
    # 获取当前x轴的范围
    x_min, x_max = ax.get_xlim()
    # 设置第二个x轴的范围（周期 = 1/频率）
    ax2.set_xlim(1/x_max, 1/x_min)
    ax2.set_xlabel('周期 (天)')
    
    # 标记一些常见的周期
    common_periods = [365.25, 180, 90, 30, 7]  # 年、半年、季度、月、周
    common_freqs = [1/p for p in common_periods]
    
    for period, freq in zip(common_periods, common_freqs):
        if freq > x_min and freq < x_max:
            ax.axvline(x=freq, color='g', linestyle='--', alpha=0.5)
            ax.text(freq, ax.get_ylim()[1]*0.9, f"{period:.0f}天", 
                    rotation=90, verticalalignment='top')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    fig.tight_layout()
    
    return fig


def apply_low_pass_filter(fft_values, frequencies, cutoff_freq):
    """
    应用低通滤波器
    
    参数:
        fft_values (numpy.ndarray): 傅立叶变换系数数组
        frequencies (numpy.ndarray): 频率数组
        cutoff_freq (float): 截止频率
    
    返回:
        numpy.ndarray: 滤波后的傅立叶变换系数数组
    """
    # 创建滤波器掩码
    # 使用高斯滤波器实现平滑过渡
    sigma = cutoff_freq / 3.0  # 控制过渡带宽
    filter_mask = np.exp(-(frequencies**2) / (2 * sigma**2))
    
    # 应用滤波器
    filtered_fft_values = fft_values * filter_mask
    
    return filtered_fft_values


def apply_high_pass_filter(fft_values, frequencies, cutoff_freq):
    """
    应用高通滤波器
    
    参数:
        fft_values (numpy.ndarray): 傅立叶变换系数数组
        frequencies (numpy.ndarray): 频率数组
        cutoff_freq (float): 截止频率
    
    返回:
        numpy.ndarray: 滤波后的傅立叶变换系数数组
    """
    # 创建滤波器掩码
    # 使用高斯滤波器的补集实现平滑过渡
    sigma = cutoff_freq / 3.0  # 控制过渡带宽
    filter_mask = 1 - np.exp(-(frequencies**2) / (2 * sigma**2))
    
    # 应用滤波器
    filtered_fft_values = fft_values * filter_mask
    
    return filtered_fft_values


def apply_band_pass_filter(fft_values, frequencies, low_cutoff_freq, high_cutoff_freq):
    """
    应用带通滤波器
    
    参数:
        fft_values (numpy.ndarray): 傅立叶变换系数数组
        frequencies (numpy.ndarray): 频率数组
        low_cutoff_freq (float): 低截止频率
        high_cutoff_freq (float): 高截止频率
    
    返回:
        numpy.ndarray: 滤波后的傅立叶变换系数数组
    """
    # 创建滤波器掩码
    # 使用两个高斯滤波器的组合实现平滑过渡的带通滤波器
    sigma_low = low_cutoff_freq / 3.0  # 控制低频过渡带宽
    sigma_high = high_cutoff_freq / 3.0  # 控制高频过渡带宽
    
    # 低通部分
    low_pass = np.exp(-(frequencies**2) / (2 * sigma_high**2))
    
    # 高通部分
    high_pass = 1 - np.exp(-(frequencies**2) / (2 * sigma_low**2))
    
    # 带通 = 低通 * 高通
    filter_mask = low_pass * high_pass
    
    # 应用滤波器
    filtered_fft_values = fft_values * filter_mask
    
    return filtered_fft_values


def inverse_fft(fft_values):
    """
    执行逆傅立叶变换
    
    参数:
        fft_values (numpy.ndarray): 傅立叶变换系数数组
    
    返回:
        numpy.ndarray: 逆变换后的时域信号
    """
    # 执行逆傅立叶变换
    inverse_values = np.fft.ifft(fft_values)
    
    # 取实部（由于输入是实数信号，输出应该也是实数）
    # 但由于数值误差，可能会有很小的虚部，所以我们取实部
    inverse_values = np.real(inverse_values)
    
    return inverse_values


def plot_filtered_results(dates, original_values, filtered_values, filter_type):
    """
    绘制原始数据和滤波后的数据进行比较
    
    参数:
        dates (numpy.ndarray): 日期数组
        original_values (numpy.ndarray): 原始数据数组
        filtered_values (numpy.ndarray): 滤波后的数据数组
        filter_type (str): 滤波器类型描述
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制原始数据
    ax.plot(dates, original_values, 'b-', alpha=0.5, linewidth=1, label='原始数据')
    
    # 绘制滤波后的数据
    ax.plot(dates, filtered_values, 'r-', linewidth=2, label=filter_type)
    
    # 添加标题和标签
    ax.set_title(f'道琼斯指数: {filter_type}')
    ax.set_xlabel('日期')
    ax.set_ylabel('指数值')
    
    # 添加图例
    ax.legend()
    
    # 格式化x轴日期标签
    fig.autofmt_xdate()
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    fig.tight_layout()
    
    return fig


def main():
    """
    主函数，执行道琼斯指数数据的傅立叶分析和滤波处理
    """
    # 数据文件路径
    data_file = "djia_data.csv"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 '{data_file}' 不存在。请确保文件位于正确的位置。")
        return
    
    # 1. 加载数据
    dates, values = load_djia_data(data_file)
    print(f"加载了 {len(values)} 个数据点")
    
    # 2. 绘制原始时间序列
    fig_time = plot_time_series(dates, values)
    plt.savefig('djia_time_series.png', dpi=300)
    plt.show()
    
    # 3. 计算傅立叶变换
    frequencies, fft_values = compute_fft(values)
    
    # 4. 绘制功率谱
    fig_spectrum = plot_power_spectrum(frequencies, fft_values)
    plt.savefig('djia_power_spectrum.png', dpi=300)
    plt.show()
    
    # 5. 应用低通滤波器（保留长期趋势）
    # 截止频率设置为每年约2个周期（相当于半年的周期）
    low_pass_cutoff = 2.0 / 365.0  # 频率单位: 每天的周期数
    low_pass_fft = apply_low_pass_filter(fft_values, frequencies, low_pass_cutoff)
    low_pass_values = inverse_fft(low_pass_fft)
    
    # 绘制低通滤波结果
    fig_low_pass = plot_filtered_results(dates, values, low_pass_values, "低通滤波 (长期趋势)")
    plt.savefig('djia_low_pass.png', dpi=300)
    plt.show()
    
    # 6. 应用高通滤波器（保留短期波动）
    # 截止频率设置为每年约12个周期（相当于一个月的周期）
    high_pass_cutoff = 12.0 / 365.0  # 频率单位: 每天的周期数
    high_pass_fft = apply_high_pass_filter(fft_values, frequencies, high_pass_cutoff)
    high_pass_values = inverse_fft(high_pass_fft)
    
    # 绘制高通滤波结果
    fig_high_pass = plot_filtered_results(dates, values, high_pass_values, "高通滤波 (短期波动)")
    plt.savefig('djia_high_pass.png', dpi=300)
    plt.show()
    
    # 7. 应用带通滤波器（保留季节性波动）
    # 截止频率设置为每年约4-12个周期（相当于1-3个月的周期）
    band_low_cutoff = 4.0 / 365.0   # 频率单位: 每天的周期数
    band_high_cutoff = 12.0 / 365.0  # 频率单位: 每天的周期数
    band_pass_fft = apply_band_pass_filter(fft_values, frequencies, band_low_cutoff, band_high_cutoff)
    band_pass_values = inverse_fft(band_pass_fft)
    
    # 绘制带通滤波结果
    fig_band_pass = plot_filtered_results(dates, values, band_pass_values, "带通滤波 (季节性波动)")
    plt.savefig('djia_band_pass.png', dpi=300)
    plt.show()
    
    # 8. 输出一些统计信息
    print("\n滤波结果统计信息:")
    print(f"原始数据均值: {np.mean(values):.2f}, 标准差: {np.std(values):.2f}")
    print(f"低通滤波后均值: {np.mean(low_pass_values):.2f}, 标准差: {np.std(low_pass_values):.2f}")
    print(f"高通滤波后均值: {np.mean(high_pass_values):.2f}, 标准差: {np.std(high_pass_values):.2f}")
    print(f"带通滤波后均值: {np.mean(band_pass_values):.2f}, 标准差: {np.std(band_pass_values):.2f}")


if __name__ == "__main__":
    main()