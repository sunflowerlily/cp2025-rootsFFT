#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 测试模块

本模块包含对太阳黑子数据周期性分析功能的单元测试。
"""

import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 添加父目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.periodicity_solution import (
from periodicity_student import (
    load_sunspot_data,
    plot_time_series as plot_sunspot_data,
    compute_fft,
    plot_power_spectrum
)


class TestPeriodicityAnalysis(unittest.TestCase):
    """太阳黑子周期性分析功能测试"""
    
    def setUp(self):
        """初始化测试数据"""
        # 创建测试信号 - 包含两个已知周期
        self.signal_length = 1000
        self.time = np.linspace(0, 10, self.signal_length)
        
        # 主周期 (约11年/132个月)
        self.primary_period = 132  
        # 次周期 (约5.5年/66个月)
        self.secondary_period = 66
        
        # 生成合成信号
        self.signal = (
            2.0 * np.sin(2 * np.pi * self.time / (self.primary_period/self.signal_length * 10)) +
            1.0 * np.sin(2 * np.pi * self.time / (self.secondary_period/self.signal_length * 10)) +
            0.5 * np.random.randn(self.signal_length)  # 添加高斯噪声
        )
        
        # 测试用的太阳黑子数据
        self.test_sunspot_data = """1749 1 1749.042 96.7 0.0 1 1
1749 2 1749.123 104.3 0.0 1 1
1749 3 1749.204 116.7 0.0 1 1"""
    
    def test_data_loading(self):
        """测试数据加载功能"""
        test_file = "test_sunspot_data.txt"
        try:
            with open(test_file, "w") as f:
                f.write(self.test_sunspot_data)
            
            # 测试数据加载
            times, counts = load_sunspot_data(test_file)
            
            # 验证返回类型
            self.assertIsInstance(times, np.ndarray)
            self.assertIsInstance(counts, np.ndarray)
            
            # 验证数据完整性
            self.assertEqual(len(times), 3)
            self.assertEqual(len(counts), 3)
            
            # 验证数据准确性
            np.testing.assert_almost_equal(times[0], 1749.042, decimal=3)
            np.testing.assert_almost_equal(counts[1], 104.3, decimal=1)
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_fft_analysis(self):
        """测试傅里叶变换分析"""
        freqs, power = compute_fft(self.signal)
        
        # 验证返回类型
        self.assertIsInstance(freqs, np.ndarray)
        self.assertIsInstance(power, np.ndarray)
        
        # 验证数组长度
        self.assertTrue(len(freqs) >= self.signal_length // 2 - 10)
        self.assertTrue(len(freqs) <= self.signal_length // 2 + 10)
        
        # 验证频率范围
        self.assertTrue(np.all(freqs >= 0))
        self.assertTrue(np.all(freqs <= 0.5))
        
        # 验证是否能检测主周期
        main_freq = 1 / self.primary_period
        freq_idx = np.argmin(np.abs(freqs - main_freq))
        self.assertGreater(power[freq_idx], np.mean(power) * 2)
    
    def test_plot_functions(self):
        """测试绘图功能"""
        # 测试太阳黑子数据绘图
        test_times = np.array([1749.042, 1749.123, 1749.204])
        test_counts = np.array([96.7, 104.3, 116.7])
        
        fig1 = plot_sunspot_data(test_times, test_counts)
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # 测试功率谱绘图
        test_freqs = np.linspace(0.01, 0.5, 100)
        test_power = np.exp(-(test_freqs - 0.1)**2 / 0.01) * 100
        
        fig2, period = plot_power_spectrum(test_freqs, test_power)
        self.assertIsInstance(fig2, plt.Figure)
        self.assertIsInstance(period, float)
        plt.close(fig2)


if __name__ == "__main__":
    unittest.main()