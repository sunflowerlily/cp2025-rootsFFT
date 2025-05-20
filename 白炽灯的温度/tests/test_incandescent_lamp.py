#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯的温度 - 测试模块

本模块包含对白炽灯效率计算实现的测试用例。
"""

import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import integrate

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生代码
try:
    import incandescent_lamp_student as incandescent_lamp
except ImportError:
    print("无法导入学生代码模块 'incandescent_lamp_student.py'")
    sys.exit(1)


class TestIncandescentLamp(unittest.TestCase):
    """测试白炽灯效率计算的实现"""
    
    def test_planck_law_points_10(self):
        """测试普朗克黑体辐射公式的实现"""
        try:
            # 测试单个波长和温度
            wavelength = 500e-9  # 500 nm
            temperature = 3000  # 3000 K
            intensity = incandescent_lamp.planck_law(wavelength, temperature)
            
            # 验证返回值是否为数值
            self.assertIsInstance(intensity, (int, float, np.number))
            
            # 验证返回值是否为正数
            self.assertGreater(intensity, 0)
            
            # 验证结果是否在合理范围内
            # 对于500nm和3000K，辐射强度应该在10^12到10^14 W/(m^2·m)范围内
            self.assertGreater(intensity, 1e12)
            self.assertLess(intensity, 1e14)
            
            # 测试数组输入
            wavelengths = np.array([400e-9, 500e-9, 600e-9])  # 400, 500, 600 nm
            intensities = incandescent_lamp.planck_law(wavelengths, temperature)
            
            # 验证返回值是否为数组
            self.assertIsInstance(intensities, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(intensities), len(wavelengths))
            
            # 验证所有值是否为正数
            self.assertTrue(np.all(intensities > 0))
            
            # 验证温度对辐射强度的影响
            # 温度越高，辐射强度应该越大
            intensity_low = incandescent_lamp.planck_law(wavelength, 2000)  # 2000 K
            intensity_high = incandescent_lamp.planck_law(wavelength, 4000)  # 4000 K
            self.assertGreater(intensity_high, intensity_low)
            
            # 验证波长对辐射强度的影响
            # 对于3000K的黑体，峰值波长约为966nm，所以500nm处的辐射强度应该小于900nm处的辐射强度
            intensity_500nm = incandescent_lamp.planck_law(500e-9, 3000)
            intensity_900nm = incandescent_lamp.planck_law(900e-9, 3000)
            self.assertLess(intensity_500nm, intensity_900nm)
            
        except NotImplementedError:
            self.fail("planck_law 函数未实现")
        except Exception as e:
            self.fail(f"planck_law 函数执行出错: {str(e)}")
    
    def test_plot_blackbody_spectrum_points_5(self):
        """测试黑体辐射谱绘图函数"""
        try:
            # 调用学生的绘图函数
            temperatures = [2000, 4000]
            fig = incandescent_lamp.plot_blackbody_spectrum(temperatures)
            
            # 验证返回值是否为Figure对象
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图形是否包含至少两条线（对应两个温度）
            ax = fig.axes[0]  # 获取第一个子图
            self.assertGreaterEqual(len(ax.lines), len(temperatures))
            
            # 验证图形是否有标题、x轴和y轴标签
            self.assertIsNotNone(ax.get_title())
            self.assertIsNotNone(ax.get_xlabel())
            self.assertIsNotNone(ax.get_ylabel())
            
            # 关闭图形，避免显示
            plt.close(fig)
            
        except NotImplementedError:
            self.fail("plot_blackbody_spectrum 函数未实现")
        except Exception as e:
            self.fail(f"plot_blackbody_spectrum 函数执行出错: {str(e)}")
    
    def test_calculate_visible_power_ratio_points_15(self):
        """测试可见光效率计算函数"""
        try:
            # 调用学生的效率计算函数
            temperature = 3000  # 3000 K
            efficiency = incandescent_lamp.calculate_visible_power_ratio(temperature)
            
            # 验证返回值是否为数值
            self.assertIsInstance(efficiency, (int, float, np.number))
            
            # 验证返回值是否在合理范围内（0到1之间）
            self.assertGreaterEqual(efficiency, 0)
            self.assertLessEqual(efficiency, 1)
            
            # 验证温度对效率的影响
            # 对于非常低的温度（如1000K），效率应该很低
            low_temp_efficiency = incandescent_lamp.calculate_visible_power_ratio(1000)
            self.assertLess(low_temp_efficiency, 0.1)  # 效率应该小于10%
            
            # 对于非常高的温度（如10000K），效率应该相对较高
            high_temp_efficiency = incandescent_lamp.calculate_visible_power_ratio(10000)
            self.assertGreater(high_temp_efficiency, 0.1)  # 效率应该大于10%
            
            # 验证最佳温度附近的效率
            # 根据维恩位移定律，黑体辐射峰值波长与温度成反比
            # 对于可见光中心波长约550nm，对应的温度约为5300K
            # 所以5000K附近的效率应该相对较高
            mid_temp_efficiency = incandescent_lamp.calculate_visible_power_ratio(5000)
            self.assertGreater(mid_temp_efficiency, low_temp_efficiency)
            
        except NotImplementedError:
            self.fail("calculate_visible_power_ratio 函数未实现")
        except Exception as e:
            self.fail(f"calculate_visible_power_ratio 函数执行出错: {str(e)}")
    
    def test_plot_efficiency_vs_temperature_points_5(self):
        """测试效率与温度关系绘图函数"""
        try:
            # 调用学生的绘图函数
            temp_range = np.linspace(1000, 10000, 10)  # 使用较少的点以加快测试速度
            fig, temperatures, efficiencies = incandescent_lamp.plot_efficiency_vs_temperature(temp_range)
            
            # 验证返回值
            self.assertIsInstance(fig, plt.Figure)
            self.assertIsInstance(temperatures, np.ndarray)
            self.assertIsInstance(efficiencies, np.ndarray)
            
            # 验证数组长度
            self.assertEqual(len(temperatures), len(temp_range))
            self.assertEqual(len(efficiencies), len(temp_range))
            
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
            self.fail("plot_efficiency_vs_temperature 函数未实现")
        except Exception as e:
            self.fail(f"plot_efficiency_vs_temperature 函数执行出错: {str(e)}")
    
    def test_golden_section_search_points_10(self):
        """测试黄金分割法函数"""
        try:
            # 定义一个简单的测试函数（抛物线，最大值在x=2处）
            def test_function(x):
                return -(x - 2)**2 + 4  # 最大值为4，位于x=2
            
            # 调用学生的黄金分割法函数
            a, b = 0, 4  # 搜索区间
            x_max, f_max = incandescent_lamp.golden_section_search(test_function, a, b)
            
            # 验证返回值是否为数值
            self.assertIsInstance(x_max, (int, float, np.number))
            self.assertIsInstance(f_max, (int, float, np.number))
            
            # 验证最大值位置是否接近真实值
            self.assertAlmostEqual(x_max, 2.0, delta=0.1)
            
            # 验证最大值是否接近真实值
            self.assertAlmostEqual(f_max, 4.0, delta=0.1)
            
            # 测试另一个函数（正弦函数，在[0,π]区间内最大值在x=π/2处）
            def sine_function(x):
                return np.sin(x)  # 最大值为1，位于x=π/2
            
            # 调用学生的黄金分割法函数
            a, b = 0, np.pi  # 搜索区间
            x_max, f_max = incandescent_lamp.golden_section_search(sine_function, a, b)
            
            # 验证最大值位置是否接近真实值
            self.assertAlmostEqual(x_max, np.pi/2, delta=0.1)
            
            # 验证最大值是否接近真实值
            self.assertAlmostEqual(f_max, 1.0, delta=0.1)
            
        except NotImplementedError:
            self.fail("golden_section_search 函数未实现")
        except Exception as e:
            self.fail(f"golden_section_search 函数执行出错: {str(e)}")
    
    def test_find_optimal_temperature_points_5(self):
        """测试最佳温度寻找函数"""
        try:
            # 调用学生的最佳温度寻找函数
            optimal_temp, optimal_efficiency = incandescent_lamp.find_optimal_temperature()
            
            # 验证返回值是否为数值
            self.assertIsInstance(optimal_temp, (int, float, np.number))
            self.assertIsInstance(optimal_efficiency, (int, float, np.number))
            
            # 验证最佳温度是否在合理范围内
            # 根据维恩位移定律和可见光范围，最佳温度应该在3000K到7000K之间
            self.assertGreaterEqual(optimal_temp, 3000)
            self.assertLessEqual(optimal_temp, 7000)
            
            # 验证最佳效率是否在合理范围内
            self.assertGreaterEqual(optimal_efficiency, 0)
            self.assertLessEqual(optimal_efficiency, 1)
            
            # 验证最佳效率是否大于低温和高温下的效率
            low_temp_efficiency = incandescent_lamp.calculate_visible_power_ratio(1000)
            high_temp_efficiency = incandescent_lamp.calculate_visible_power_ratio(10000)
            self.assertGreater(optimal_efficiency, low_temp_efficiency)
            self.assertGreater(optimal_efficiency, high_temp_efficiency)
            
        except NotImplementedError:
            self.fail("find_optimal_temperature 函数未实现")
        except Exception as e:
            self.fail(f"find_optimal_temperature 函数执行出错: {str(e)}")


if __name__ == "__main__":
    unittest.main()