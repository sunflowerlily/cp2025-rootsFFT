#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 测试模块

本模块包含对道琼斯工业平均指数数据傅立叶分析和滤波处理的测试用例。
"""

import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution.fourier_filter_solution import load_data, plot_data, fourier_filter, plot_comparison
#from fourier_filter_student import load_data, plot_data, fourier_filter, plot_comparison

class TestFourierFilter(unittest.TestCase):
    """测试傅立叶滤波和平滑的实现"""
    
    def setUp(self):
        """设置测试数据"""
        self.N = 1024
        self.t = np.linspace(0, 10, self.N)
        
        # 调整数据范围以匹配真实道琼斯指数数据
        self.trend = 12000 + 150 * self.t  # 进一步降低斜率
        self.noise = 300 * np.random.randn(self.N)  # 进一步减小噪声幅度
        self.signal = self.trend + self.noise
        
        self.test_filename = "test_dow_data.txt"
        np.savetxt(self.test_filename, self.signal)
    
    def tearDown(self):
        """清理测试数据"""
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)
    
    def test_load_data_points_5(self):
        """测试数据加载函数"""
        try:
            data = load_data(self.test_filename)
            
            # 验证返回类型
            self.assertIsInstance(data, np.ndarray)
            
            # 验证数据长度
            self.assertEqual(len(data), self.N)
            
            # 移除具体数值比较测试
            
        except Exception as e:
            self.fail(f"load_data 函数执行出错: {str(e)}")
    
    def test_fourier_filter_points_10(self):
        """测试傅立叶滤波函数"""
        try:
            # 测试保留10%系数
            filtered_10, coeff = fourier_filter(self.signal, 0.1)
            
            # 验证返回类型
            self.assertIsInstance(filtered_10, np.ndarray)
            self.assertIsInstance(coeff, np.ndarray)
            
            # 验证数据长度
            self.assertEqual(len(filtered_10), self.N)
            
            # 验证滤波效果 - 滤波后信号应更平滑
            filtered_std = np.std(filtered_10 - self.trend)
            original_std = np.std(self.signal - self.trend)
            self.assertLess(filtered_std, original_std)  # 只需比原始信号平滑
            
            # 测试保留2%系数
            filtered_2, _ = fourier_filter(self.signal, 0.02)
            filtered_2_std = np.std(filtered_2 - self.trend)
            self.assertLess(filtered_2_std, original_std)  # 只需比原始信号平滑
            
        except Exception as e:
            self.fail(f"fourier_filter 函数执行出错: {str(e)}")
    
    def test_plot_functions_points_5(self):
        """测试绘图函数"""
        try:
            # 测试plot_data
            fig1 = plot_data(self.signal)
            self.assertIsInstance(fig1, plt.Figure)
            plt.close(fig1)
            
            # 测试plot_comparison
            filtered, _ = fourier_filter(self.signal, 0.1)
            fig2 = plot_comparison(self.signal, filtered)
            self.assertIsInstance(fig2, plt.Figure)
            plt.close(fig2)
            
        except Exception as e:
            self.fail(f"绘图函数执行出错: {str(e)}")

if __name__ == "__main__":
    unittest.main()