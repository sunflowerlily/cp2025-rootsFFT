#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日点 - 测试模块

本模块包含对地球-月球系统L1拉格朗日点位置计算实现的测试用例。
"""

import unittest
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.lagrange_solution import (
from lagrange_student import (
    lagrange_equation,
    lagrange_equation_derivative,
    newton_method,
    secant_method,
    G, M, m, R, omega
)

class TestLagrangeEquation(unittest.TestCase):
    """测试L1点位置方程及其导数"""
    
    def setUp(self):
        self.r_expected = 3.263e8  # 预期L1点位置 (m)
        self.tol = 1e-4  # 相对容差
    
    def test_lagrange_equation_at_L1(self):
        """测试L1点位置方程在预期解处为零"""
        result = lagrange_equation(self.r_expected)
        self.assertAlmostEqual(result, 0.0, delta=1e-3)
    
    def test_lagrange_equation_derivative_at_L1(self):
        """测试L1点位置方程导数在预期解处的值"""
        result = lagrange_equation_derivative(self.r_expected)
        self.assertLess(abs(result), 1e-10)  # 导数应接近零
    
    def test_equation_values_at_boundaries(self):
        """测试方程在边界点的符号变化"""
        r_close_to_earth = 1e7  # 靠近地球
        r_close_to_moon = R - 1e7  # 靠近月球
        
        # 方程在边界点应异号
        f1 = lagrange_equation(r_close_to_earth)
        f2 = lagrange_equation(r_close_to_moon)
        self.assertTrue(f1 * f2 < 0)

class TestNewtonMethod(unittest.TestCase):
    """测试牛顿法求解"""
    
    def test_newton_method_convergence(self):
        """测试牛顿法收敛到L1点"""
        r0 = 3.5e8  # 初始猜测
        r, iterations, converged = newton_method(
            lagrange_equation, 
            lagrange_equation_derivative, 
            r0
        )
        self.assertTrue(converged)
        self.assertLessEqual(iterations, 10)
        self.assertAlmostEqual(r, 3.263e8, delta=1e6)  # 允许1,000km误差
    
    def test_newton_method_precision(self):
        """测试牛顿法求解精度"""
        r0 = 3.5e8
        r, _, _ = newton_method(
            lagrange_equation, 
            lagrange_equation_derivative, 
            r0,
            tol=1e-10
        )
        # 验证方程在解处接近零
        self.assertAlmostEqual(lagrange_equation(r), 0.0, delta=1e-10)

class TestSecantMethod(unittest.TestCase):
    """测试弦截法求解"""
    
    def test_secant_method_convergence(self):
        """测试弦截法收敛到L1点"""
        a, b = 3.0e8, 3.5e8  # 初始区间
        r, iterations, converged = secant_method(lagrange_equation, a, b)
        self.assertTrue(converged)
        self.assertLessEqual(iterations, 15)
        self.assertAlmostEqual(r, 3.263e8, delta=1e6)  # 允许1,000km误差
    
    def test_secant_method_precision(self):
        """测试弦截法求解精度"""
        a, b = 3.0e8, 3.5e8
        r, _, _ = secant_method(
            lagrange_equation, 
            a, b,
            tol=1e-10
        )
        # 验证方程在解处接近零
        self.assertAlmostEqual(lagrange_equation(r), 0.0, delta=1e-10)

if __name__ == "__main__":
    unittest.main()