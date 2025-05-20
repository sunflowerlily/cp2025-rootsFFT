#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 测试模块

本模块包含对方势阱能级计算实现的测试用例。
"""

import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生代码
try:
    import schrodinger_student as schrodinger
except ImportError:
    print("无法导入学生代码模块 'schrodinger_student.py'")
    sys.exit(1)


class TestSchrodingerEquation(unittest.TestCase):
    """测试方势阱能级计算的实现"""
    
    def setUp(self):
        """设置测试参数"""
        self.V = 20.0  # 势阱高度 (eV)
        self.w = 1e-9  # 势阱宽度 (m)
        self.m = schrodinger.ELECTRON_MASS  # 粒子质量 (kg)
        self.E_values = np.linspace(0.001, 19.999, 100)  # 能量范围 (eV)
        self.reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]  # 参考能级值
        self.tolerance = 0.01  # 能级计算的容差 (eV)
    
    def test_calculate_y_values_points_5(self):
        """测试y值计算函数在特定点的正确性"""
        # 选择几个特定的能量点进行测试
        test_energies = np.array([0.5, 5.0, 10.0])
        
        # 计算学生代码的输出
        try:
            y1, y2, y3 = schrodinger.calculate_y_values(test_energies, self.V, self.w, self.m)
            
            # 验证输出是否为numpy数组
            self.assertIsInstance(y1, np.ndarray)
            self.assertIsInstance(y2, np.ndarray)
            self.assertIsInstance(y3, np.ndarray)
            
            # 验证输出数组长度
            self.assertEqual(len(y1), len(test_energies))
            self.assertEqual(len(y2), len(test_energies))
            self.assertEqual(len(y3), len(test_energies))
            
            # 验证y值的符号和大致范围
            # 这里只做基本检查，不验证精确值
            self.assertTrue(np.all(np.isfinite(y1)))
            self.assertTrue(np.all(np.isfinite(y2)))
            self.assertTrue(np.all(np.isfinite(y3)))
            
            # y2应该是正值，y3应该是负值
            self.assertTrue(np.all(y2 > 0))
            self.assertTrue(np.all(y3 < 0))
            
        except NotImplementedError:
            self.fail("calculate_y_values 函数未实现")
        except Exception as e:
            self.fail(f"calculate_y_values 函数执行出错: {str(e)}")
    
    def test_plot_energy_functions_points_5(self):
        """测试绘图函数的基本功能"""
        # 生成测试数据
        E_test = np.linspace(0.1, 19.9, 10)
        y1_test = np.sin(E_test)  # 使用简单函数作为测试数据
        y2_test = np.cos(E_test)
        y3_test = -np.cos(E_test)
        
        try:
            # 调用学生的绘图函数
            fig = schrodinger.plot_energy_functions(E_test, y1_test, y2_test, y3_test)
            
            # 验证返回值是否为Figure对象
            self.assertIsInstance(fig, plt.Figure)
            
            # 验证图形是否包含至少3条线（对应三个函数）
            ax = fig.axes[0]  # 获取第一个子图
            self.assertGreaterEqual(len(ax.lines), 3)
            
            # 验证图形是否有标题、x轴和y轴标签
            self.assertIsNotNone(ax.get_title())
            self.assertIsNotNone(ax.get_xlabel())
            self.assertIsNotNone(ax.get_ylabel())
            
            # 验证图例是否存在
            self.assertTrue(ax.get_legend() is not None)
            
            # 关闭图形，避免显示
            plt.close(fig)
            
        except NotImplementedError:
            self.fail("plot_energy_functions 函数未实现")
        except Exception as e:
            self.fail(f"plot_energy_functions 函数执行出错: {str(e)}")
    
    def test_find_energy_level_bisection_ground_state_points_5(self):
        """测试二分法求解基态能级"""
        try:
            # 计算基态能级
            energy_level = schrodinger.find_energy_level_bisection(0, self.V, self.w, self.m)
            
            # 验证能级值是否在合理范围内
            self.assertGreater(energy_level, 0)
            self.assertLess(energy_level, self.V)
            
            # 验证与参考值的误差是否在容差范围内
            reference = self.reference_levels[0]
            self.assertAlmostEqual(energy_level, reference, delta=self.tolerance)
            
        except NotImplementedError:
            self.fail("find_energy_level_bisection 函数未实现")
        except Exception as e:
            self.fail(f"find_energy_level_bisection 函数执行出错: {str(e)}")
    
    def test_find_energy_level_bisection_excited_states_points_10(self):
        """测试二分法求解激发态能级"""
        try:
            # 计算前6个能级
            energy_levels = []
            for n in range(6):
                energy = schrodinger.find_energy_level_bisection(n, self.V, self.w, self.m)
                energy_levels.append(energy)
            
            # 验证能级是否递增
            for i in range(1, len(energy_levels)):
                self.assertGreater(energy_levels[i], energy_levels[i-1])
            
            # 验证与参考值的误差是否在容差范围内
            for i, (calc, ref) in enumerate(zip(energy_levels, self.reference_levels)):
                self.assertAlmostEqual(calc, ref, delta=self.tolerance,
                                      msg=f"能级 {i} 计算值 {calc:.3f} 与参考值 {ref:.3f} 相差过大")
            
        except NotImplementedError:
            self.fail("find_energy_level_bisection 函数未实现")
        except Exception as e:
            self.fail(f"find_energy_level_bisection 函数执行出错: {str(e)}")


if __name__ == "__main__":
    unittest.main()