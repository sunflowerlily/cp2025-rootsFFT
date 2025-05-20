#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探测周期性 - 测试模块

本模块包含对太阳黑子数据周期性分析实现的测试用例。
"""

import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生代码
try:
    import periodicity_student as periodicity
except ImportError:
    print("无法导入学生代码模块 'periodicity_student.py'")
    sys.exit(1)


class TestPeriodicity(unittest.TestCase):
    """测试太阳黑子数据周期性分析的实现"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建一个简单的测试信号，包含已知周期
        self.N = 1000  # 信号长度
        self.t = np.linspace(0, 10, self.N)  # 时间数组
        
        # 创建一个包含多个周期成分的信号
        period1 = 100  # 100个点的周期
        period2 = 50   # 50个点的周期
        amplitude1 = 2.0
        amplitude2 = 1.0
        
        self.signal = amplitude1 * np.sin(2 * np.pi * self.t / (period1 / self.N * 10)) + \
                      amplitude2 * np.sin(2 * np.pi * self.t / (period2 / self.N * 10)) + \
                      0.5 * np.random.randn(self.N)  # 添加噪声
        
        # 预期的主要周期（以点数为单位）
        self.expected_period = period1
    
    def test_load_sunspot_data_points_5(self):
        """测试太阳黑子数据加载函数"""
        # 创建一个简单的测试数据文件
        test_data = """1749 1 1749.042 96.7 0.0 1 1
1749 2 1749.123 104.3 0.0 1 1
1749 3 1749.204 116.7 0.0 1 1"""
        test_filename = "test_sunspot_data.txt"
        
        with open(test_filename, "w") as f:
            f.write(test_data)
        
        try:
            # 调用学生的数据加载函数
            times, sunspots = periodicity.load_sunspot_data(test_filename)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(times, np.ndarray)
            self.assertIsInstance(sunspots, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(times), 3)
            self.assertEqual(len(sunspots), 3)
            
            # 验证数据值
            np.testing.assert_almost_equal(times[0], 1749.042, decimal=3)
            np.testing.assert_almost_equal(sunspots[0], 96.7, decimal=1)
            
        except NotImplementedError:
            self.fail("load_sunspot_data 函数未实现")
        except Exception as e:
            self.fail(f"load_sunspot_data 函数执行出错: {str(e)}")
        finally:
            # 清理测试文件
            if os.path.exists(test_filename):
                os.remove(test_filename)
    
    def test_plot_sunspot_data_points_5(self):
        """测试太阳黑子数据绘图函数"""
        # 创建测试数据
        times = np.array([1749.042, 1749.123, 1749.204, 1749.288, 1749.371])
        sunspots = np.array([96.7, 104.3, 116.7, 92.5, 141.3])
        
        try:
            # 调用学生的绘图函数
            fig = periodicity.plot_sunspot_data(times, sunspots)
            
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
            self.fail("plot_sunspot_data 函数未实现")
        except Exception as e:
            self.fail(f"plot_sunspot_data 函数执行出错: {str(e)}")
    
    def test_compute_fft_points_10(self):
        """测试傅里叶变换计算函数"""
        try:
            # 调用学生的傅里叶变换函数
            frequencies, power_spectrum = periodicity.compute_fft(self.signal)
            
            # 验证返回值是否为numpy数组
            self.assertIsInstance(frequencies, np.ndarray)
            self.assertIsInstance(power_spectrum, np.ndarray)
            
            # 验证数组长度（应该是信号长度的一半左右，因为我们只关注正频率部分）
            self.assertGreaterEqual(len(frequencies), self.N // 2 - 10)
            self.assertLessEqual(len(frequencies), self.N // 2 + 10)
            self.assertEqual(len(frequencies), len(power_spectrum))
            
            # 验证频率范围（应该在0到0.5之间，因为采样率是1）
            self.assertGreater(np.min(frequencies), 0)  # 排除零频率
            self.assertLessEqual(np.max(frequencies), 0.5)
            
            # 验证功率谱是否为非负值
            self.assertTrue(np.all(power_spectrum >= 0))
            
            # 验证是否能检测到主要周期
            # 找出功率谱中的峰值
            peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.5)
            
            if len(peaks) > 0:
                # 获取最大峰值对应的频率
                max_peak_idx = peaks[np.argmax(power_spectrum[peaks])]
                max_peak_freq = frequencies[max_peak_idx]
                
                # 计算对应的周期（以点数为单位）
                detected_period = 1 / max_peak_freq
                
                # 验证检测到的周期是否接近预期周期
                # 允许10%的误差
                rel_error = abs(detected_period - self.expected_period) / self.expected_period
                self.assertLess(rel_error, 0.1, 
                               f"检测到的周期 {detected_period:.1f} 与预期周期 {self.expected_period} 相差过大")
            
        except NotImplementedError:
            self.fail("compute_fft 函数未实现")
        except Exception as e:
            self.fail(f"compute_fft 函数执行出错: {str(e)}")
    
    def test_plot_power_spectrum_points_5(self):
        """测试功率谱绘图函数"""
        # 创建测试数据
        frequencies = np.linspace(0.01, 0.5, 100)
        power_spectrum = np.zeros_like(frequencies)
        
        # 在特定频率处添加峰值
        peak_idx = 20  # 对应频率约为0.1
        power_spectrum[peak_idx] = 100
        power_spectrum[peak_idx-1] = 50
        power_spectrum[peak_idx+1] = 50
        
        # 添加一些噪声
        power_spectrum += np.random.rand(len(power_spectrum)) * 10
        
        try:
            # 调用学生的功率谱绘图函数
            fig, main_period = periodicity.plot_power_spectrum(frequencies, power_spectrum)
            
            # 验证返回值
            self.assertIsInstance(fig, plt.Figure)
            self.assertIsInstance(main_period, (int, float))
            
            # 验证图形是否包含至少一条线
            ax = fig.axes[0]  # 获取第一个子图
            self.assertGreaterEqual(len(ax.lines), 1)
            
            # 验证图形是否有标题、x轴和y轴标签
            self.assertIsNotNone(ax.get_title())
            self.assertIsNotNone(ax.get_xlabel())
            self.assertIsNotNone(ax.get_ylabel())
            
            # 验证检测到的主要周期是否合理
            expected_period = 1 / frequencies[peak_idx]
            rel_error = abs(main_period - expected_period) / expected_period
            self.assertLess(rel_error, 0.2, 
                           f"检测到的周期 {main_period:.1f} 与预期周期 {expected_period:.1f} 相差过大")
            
            # 关闭图形，避免显示
            plt.close(fig)
            
        except NotImplementedError:
            self.fail("plot_power_spectrum 函数未实现")
        except Exception as e:
            self.fail(f"plot_power_spectrum 函数执行出错: {str(e)}")


if __name__ == "__main__":
    unittest.main()