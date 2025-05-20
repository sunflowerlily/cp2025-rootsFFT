#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 测试模块

本模块包含对道琼斯工业平均指数数据傅立叶分析和滤波处理实现的测试用例。
"""

import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生代码
try:
    import fourier_filter_student as fourier_filter
except ImportError:
    print("无法导入学生代码模块 'fourier_filter_student.py'")
    sys.exit(1)


class TestFourierFilter(unittest.TestCase):
    """测试傅立叶滤波和平滑的实现"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建一个简单的测试信号，包含多个频率成分
        self.N = 1000  # 信号长度
        self.t = np.linspace(0, 10, self.N)  # 时间数组
        
        # 创建测试信号：包含低频趋势、中频季节性和高频噪声
        trend = 100 + 5 * self.t  # 线性趋势
        seasonal = 20 * np.sin(2 * np.pi * self.t)  # 低频季节性 (周期=1)
        cycles = 10 * np.sin(2 * np.pi * 5 * self.t)  # 中频周期 (周期=0.2)
        noise = 5 * np.random.randn(self.N)  # 高频噪声
        
        self.signal = trend + seasonal + cycles + noise
        
        # 创建日期数组（从2000-01-01开始的连续日期）
        start_date = datetime(2000, 1, 1)
        self.dates = np.array([start_date + timedelta(days=i) for i in range(self.N)])
        
        # 创建测试数据文件
        self.test_filename = "test_djia_data.csv"
        df = pd.DataFrame({
            'Date': self.dates,
            'Close': self.signal
        })
        df.to_csv(self.test_filename, index=False)
    
    def tearDown(self):
        """清理测试数据"""
        # 删除测试数据文件
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)
    
    def test_load_djia_data_points_5(self):
        """测试数据加载函数"""
        try:
            # 调用学生的数据加载函数
            dates, values = fourier_filter.load_djia_data(self.test_filename)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(dates, np.ndarray)
            self.assertIsInstance(values, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(dates), self.N)
            self.assertEqual(len(values), self.N)
            
            # 验证数据值（检查第一个和最后一个值）
            np.testing.assert_almost_equal(values[0], self.signal[0], decimal=1)
            np.testing.assert_almost_equal(values[-1], self.signal[-1], decimal=1)
            
        except NotImplementedError:
            self.fail("load_djia_data 函数未实现")
        except Exception as e:
            self.fail(f"load_djia_data 函数执行出错: {str(e)}")
    
    def test_plot_time_series_points_5(self):
        """测试时间序列绘图函数"""
        try:
            # 调用学生的绘图函数
            fig = fourier_filter.plot_time_series(self.dates, self.signal)
            
            # 验证返回值是否为Figure对象
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图形是否包含至少一条线
            ax = fig.axes[0]  # 获取第一个子图
            self.assertGreaterEqual(len(ax.lines), 1)
            
            # 验证图形是否有标题、x轴和y轴标签
            self.assertIsNotNone(ax.get_title())
            self.assertIsNotNone(ax.get_xlabel())
            self.assertIsNotNone(ax.get_ylabel())
            
            # 关闭图形，避免显示
            plt.close(fig)
            
        except NotImplementedError:
            self.fail("plot_time_series 函数未实现")
        except Exception as e:
            self.fail(f"plot_time_series 函数执行出错: {str(e)}")
    
    def test_compute_fft_points_10(self):
        """测试傅立叶变换计算函数"""
        try:
            # 调用学生的傅立叶变换函数
            frequencies, fft_values = fourier_filter.compute_fft(self.signal)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(frequencies, np.ndarray)
            self.assertIsInstance(fft_values, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(frequencies), self.N)
            self.assertEqual(len(fft_values), self.N)
            
            # 验证频率范围（应该在-0.5到0.5之间，因为采样率是1）
            self.assertGreaterEqual(np.min(frequencies), -0.5)
            self.assertLessEqual(np.max(frequencies), 0.5)
            
            # 验证傅立叶变换的基本特性
            # 1. 直流分量（零频率）应该接近信号的平均值
            dc_index = np.argmin(np.abs(frequencies))
            dc_value = np.abs(fft_values[dc_index]) / self.N
            signal_mean = np.mean(self.signal)
            self.assertAlmostEqual(dc_value, signal_mean, delta=signal_mean*0.1)
            
            # 2. 对于实数信号，傅立叶变换应该具有共轭对称性
            # 即 X(-f) = X*(f)，其中 X* 表示 X 的共轭
            for i in range(1, self.N // 2):
                neg_idx = self.N - i
                self.assertAlmostEqual(fft_values[i].real, fft_values[neg_idx].real, delta=1e-10)
                self.assertAlmostEqual(fft_values[i].imag, -fft_values[neg_idx].imag, delta=1e-10)
            
        except NotImplementedError:
            self.fail("compute_fft 函数未实现")
        except Exception as e:
            self.fail(f"compute_fft 函数执行出错: {str(e)}")
    
    def test_plot_power_spectrum_points_5(self):
        """测试功率谱绘图函数"""
        try:
            # 先计算傅立叶变换
            frequencies, fft_values = fourier_filter.compute_fft(self.signal)
            
            # 调用学生的功率谱绘图函数
            fig = fourier_filter.plot_power_spectrum(frequencies, fft_values)
            
            # 验证返回值是否为Figure对象
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图形是否包含至少一条线
            ax = fig.axes[0]  # 获取第一个子图
            self.assertGreaterEqual(len(ax.lines), 1)
            
            # 验证图形是否有标题、x轴和y轴标签
            self.assertIsNotNone(ax.get_title())
            self.assertIsNotNone(ax.get_xlabel())
            self.assertIsNotNone(ax.get_ylabel())
            
            # 关闭图形，避免显示
            plt.close(fig)
            
        except NotImplementedError:
            self.fail("plot_power_spectrum 函数未实现")
        except Exception as e:
            self.fail(f"plot_power_spectrum 函数执行出错: {str(e)}")
    
    def test_apply_low_pass_filter_points_10(self):
        """测试低通滤波器函数"""
        try:
            # 先计算傅立叶变换
            frequencies, fft_values = fourier_filter.compute_fft(self.signal)
            
            # 设置截止频率（保留低频成分）
            cutoff_freq = 0.1  # 频率单位: 每采样点的周期数
            
            # 调用学生的低通滤波器函数
            filtered_fft = fourier_filter.apply_low_pass_filter(fft_values, frequencies, cutoff_freq)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(filtered_fft, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(filtered_fft), self.N)
            
            # 验证滤波器的基本特性
            # 1. 低频成分（|f| < cutoff_freq）应该基本保留
            low_freq_idx = np.abs(frequencies) < cutoff_freq
            for i in np.where(low_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertGreaterEqual(ratio, 0.5)  # 至少保留50%的幅度
            
            # 2. 高频成分（|f| > 2*cutoff_freq）应该基本被抑制
            high_freq_idx = np.abs(frequencies) > 2 * cutoff_freq
            for i in np.where(high_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertLessEqual(ratio, 0.5)  # 至多保留50%的幅度
            
        except NotImplementedError:
            self.fail("apply_low_pass_filter 函数未实现")
        except Exception as e:
            self.fail(f"apply_low_pass_filter 函数执行出错: {str(e)}")
    
    def test_apply_high_pass_filter_points_10(self):
        """测试高通滤波器函数"""
        try:
            # 先计算傅立叶变换
            frequencies, fft_values = fourier_filter.compute_fft(self.signal)
            
            # 设置截止频率（保留高频成分）
            cutoff_freq = 0.2  # 频率单位: 每采样点的周期数
            
            # 调用学生的高通滤波器函数
            filtered_fft = fourier_filter.apply_high_pass_filter(fft_values, frequencies, cutoff_freq)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(filtered_fft, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(filtered_fft), self.N)
            
            # 验证滤波器的基本特性
            # 1. 高频成分（|f| > cutoff_freq）应该基本保留
            high_freq_idx = np.abs(frequencies) > cutoff_freq
            for i in np.where(high_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertGreaterEqual(ratio, 0.5)  # 至少保留50%的幅度
            
            # 2. 低频成分（|f| < 0.5*cutoff_freq）应该基本被抑制
            low_freq_idx = np.abs(frequencies) < 0.5 * cutoff_freq
            for i in np.where(low_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertLessEqual(ratio, 0.5)  # 至多保留50%的幅度
            
        except NotImplementedError:
            self.fail("apply_high_pass_filter 函数未实现")
        except Exception as e:
            self.fail(f"apply_high_pass_filter 函数执行出错: {str(e)}")
    
    def test_apply_band_pass_filter_points_5(self):
        """测试带通滤波器函数"""
        try:
            # 先计算傅立叶变换
            frequencies, fft_values = fourier_filter.compute_fft(self.signal)
            
            # 设置截止频率（保留中频成分）
            low_cutoff_freq = 0.05  # 频率单位: 每采样点的周期数
            high_cutoff_freq = 0.15  # 频率单位: 每采样点的周期数
            
            # 调用学生的带通滤波器函数
            filtered_fft = fourier_filter.apply_band_pass_filter(fft_values, frequencies, low_cutoff_freq, high_cutoff_freq)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(filtered_fft, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(filtered_fft), self.N)
            
            # 验证滤波器的基本特性
            # 1. 中频成分（low_cutoff_freq < |f| < high_cutoff_freq）应该基本保留
            mid_freq_idx = (np.abs(frequencies) > low_cutoff_freq) & (np.abs(frequencies) < high_cutoff_freq)
            for i in np.where(mid_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertGreaterEqual(ratio, 0.5)  # 至少保留50%的幅度
            
            # 2. 低频成分（|f| < 0.5*low_cutoff_freq）应该基本被抑制
            low_freq_idx = np.abs(frequencies) < 0.5 * low_cutoff_freq
            for i in np.where(low_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertLessEqual(ratio, 0.5)  # 至多保留50%的幅度
            
            # 3. 高频成分（|f| > 2*high_cutoff_freq）应该基本被抑制
            high_freq_idx = np.abs(frequencies) > 2 * high_cutoff_freq
            for i in np.where(high_freq_idx)[0]:
                ratio = np.abs(filtered_fft[i]) / (np.abs(fft_values[i]) + 1e-10)
                self.assertLessEqual(ratio, 0.5)  # 至多保留50%的幅度
            
        except NotImplementedError:
            self.fail("apply_band_pass_filter 函数未实现")
        except Exception as e:
            self.fail(f"apply_band_pass_filter 函数执行出错: {str(e)}")
    
    def test_inverse_fft_points_5(self):
        """测试逆傅立叶变换函数"""
        try:
            # 先计算傅立叶变换
            frequencies, fft_values = fourier_filter.compute_fft(self.signal)
            
            # 调用学生的逆傅立叶变换函数
            inverse_values = fourier_filter.inverse_fft(fft_values)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(inverse_values, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(inverse_values), self.N)
            
            # 验证逆变换后的信号是否接近原始信号
            # 由于数值误差，我们允许一定的误差范围
            np.testing.assert_allclose(inverse_values, self.signal, rtol=1e-10, atol=1e-10)
            
            # 验证逆变换后的信号是否为实数
            self.assertTrue(np.isreal(inverse_values).all())
            
        except NotImplementedError:
            self.fail("inverse_fft 函数未实现")
        except Exception as e:
            self.fail(f"inverse_fft 函数执行出错: {str(e)}")
    
    def test_plot_filtered_results_points_5(self):
        """测试滤波结果比较绘图函数"""
        try:
            # 创建一个简单的滤波后信号（原始信号的移动平均）
            window_size = 20
            filtered_signal = np.convolve(self.signal, np.ones(window_size)/window_size, mode='same')
            
            # 调用学生的滤波结果比较绘图函数
            fig = fourier_filter.plot_filtered_results(self.dates, self.signal, filtered_signal, "测试滤波")
            
            # 验证返回值是否为Figure对象
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图形是否包含至少两条线（原始数据和滤波后的数据）
            ax = fig.axes[0]  # 获取第一个子图
            self.assertGreaterEqual(len(ax.lines), 2)
            
            # 验证图形是否有标题、x轴和y轴标签
            self.assertIsNotNone(ax.get_title())
            self.assertIsNotNone(ax.get_xlabel())
            self.assertIsNotNone(ax.get_ylabel())
            
            # 验证图形是否有图例
            self.assertIsNotNone(ax.get_legend())
            
            # 关闭图形，避免显示
            plt.close(fig)
            
        except NotImplementedError:
            self.fail("plot_filtered_results 函数未实现")
        except Exception as e:
            self.fail(f"plot_filtered_results 函数执行出错: {str(e)}")


if __name__ == "__main__":
    unittest.main()