#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日点 - 测试模块

本模块包含对地球-月球系统L1拉格朗日点位置计算实现的测试用例。
"""

import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import optimize

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入学生代码
try:
    import lagrange_student as lagrange
except ImportError:
    print("无法导入学生代码模块 'lagrange_student.py'")
    sys.exit(1)


class TestLagrangePoint(unittest.TestCase):
    """测试L1拉格朗日点位置计算的实现"""
    
    def setUp(self):
        """设置测试参数"""
        # 使用scipy.optimize.fsolve计算参考解
        self.reference_solution = optimize.fsolve(lagrange.lagrange_equation, 3.5e8)[0]
        self.tolerance = 1e-4  # 相对误差容差
    
    def test_lagrange_equation_points_5(self):
        """测试L1点位置方程在特定点的正确性"""
        # 测试点：地月距离的50%、75%和90%处
        test_points = [0.5 * lagrange.R, 0.75 * lagrange.R, 0.9 * lagrange.R]
        
        try:
            # 计算方程值
            equation_values = [lagrange.lagrange_equation(r) for r in test_points]
            
            # 验证方程值是否为浮点数
            for value in equation_values:
                self.assertIsInstance(value, float)
            
            # 验证方程在L1点附近的符号变化
            # 在L1点左侧应为正值，右侧应为负值
            left_point = 0.85 * self.reference_solution
            right_point = 1.15 * self.reference_solution
            left_value = lagrange.lagrange_equation(left_point)
            right_value = lagrange.lagrange_equation(right_point)
            
            # 验证符号变化，表明在这之间有零点
            self.assertGreater(left_value, 0)
            self.assertLess(right_value, 0)
            
        except NotImplementedError:
            self.fail("lagrange_equation 函数未实现")
        except Exception as e:
            self.fail(f"lagrange_equation 函数执行出错: {str(e)}")
    
    def test_lagrange_equation_derivative_points_5(self):
        """测试L1点位置方程导数在特定点的正确性"""
        # 测试点：地月距离的75%和90%处
        test_points = [0.75 * lagrange.R, 0.9 * lagrange.R]
        
        try:
            # 计算导数值
            derivative_values = [lagrange.lagrange_equation_derivative(r) for r in test_points]
            
            # 验证导数值是否为浮点数
            for value in derivative_values:
                self.assertIsInstance(value, float)
            
            # 使用数值微分验证导数的近似正确性
            h = 1e-6  # 微小步长
            for r in test_points:
                # 数值导数：中心差分
                numerical_derivative = (lagrange.lagrange_equation(r + h) - lagrange.lagrange_equation(r - h)) / (2 * h)
                analytical_derivative = lagrange.lagrange_equation_derivative(r)
                
                # 验证解析导数与数值导数的相对误差
                rel_error = abs(analytical_derivative - numerical_derivative) / abs(numerical_derivative)
                self.assertLess(rel_error, 0.01)  # 允许1%的相对误差
            
        except NotImplementedError:
            self.fail("lagrange_equation_derivative 函数未实现")
        except Exception as e:
            self.fail(f"lagrange_equation_derivative 函数执行出错: {str(e)}")
    
    def test_newton_method_points_5(self):
        """测试牛顿法求解L1点位置"""
        try:
            # 使用牛顿法求解
            r0 = 3.5e8  # 初始猜测值
            r_newton, iterations, converged = lagrange.newton_method(
                lagrange.lagrange_equation, lagrange.lagrange_equation_derivative, r0)
            
            # 验证是否收敛
            self.assertTrue(converged, "牛顿法未收敛")
            
            # 验证迭代次数是否合理
            self.assertLess(iterations, 20, "牛顿法迭代次数过多")
            
            # 验证解的精度
            rel_error = abs(r_newton - self.reference_solution) / self.reference_solution
            self.assertLess(rel_error, self.tolerance, 
                           f"牛顿法解的相对误差 {rel_error:.8f} 超过容差 {self.tolerance}")
            
        except NotImplementedError:
            self.fail("newton_method 函数未实现")
        except Exception as e:
            self.fail(f"newton_method 函数执行出错: {str(e)}")
    
    def test_secant_method_points_5(self):
        """测试弦截法求解L1点位置"""
        try:
            # 使用弦截法求解
            a, b = 3.2e8, 3.7e8  # 初始区间
            r_secant, iterations, converged = lagrange.secant_method(
                lagrange.lagrange_equation, a, b)
            
            # 验证是否收敛
            self.assertTrue(converged, "弦截法未收敛")
            
            # 验证迭代次数是否合理
            self.assertLess(iterations, 20, "弦截法迭代次数过多")
            
            # 验证解的精度
            rel_error = abs(r_secant - self.reference_solution) / self.reference_solution
            self.assertLess(rel_error, self.tolerance, 
                           f"弦截法解的相对误差 {rel_error:.8f} 超过容差 {self.tolerance}")
            
        except NotImplementedError:
            self.fail("secant_method 函数未实现")
        except Exception as e:
            self.fail(f"secant_method 函数执行出错: {str(e)}")
    
    def test_plot_lagrange_equation_points_5(self):
        """测试L1点位置方程绘图函数"""
        try:
            # 调用绘图函数
            r_min, r_max = 3.0e8, 3.8e8
            fig = lagrange.plot_lagrange_equation(r_min, r_max, num_points=100)
            
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
            self.fail("plot_lagrange_equation 函数未实现")
        except Exception as e:
            self.fail(f"plot_lagrange_equation 函数执行出错: {str(e)}")


if __name__ == "__main__":
    unittest.main()