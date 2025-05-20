#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化 - 测试代码
"""
import unittest
import numpy as np
from scipy.integrate import quad

# 导入解决方案模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solution.incandescent_lamp_solution import (
from incandescent_lamp_student import (
    planck_law,
    calculate_visible_power_ratio,
    find_optimal_temperature,
    H, C, K_B,
    VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX
)

class TestPlanckLaw(unittest.TestCase):
    """测试普朗克黑体辐射公式"""
    
    def test_planck_law_at_500nm_3000K(self):
        """测试500nm波长在3000K温度下的辐射强度"""
        wavelength = 500e-9  # 500 nm
        temp = 3000  # 3000 K
        expected = 2.0 * H * C**2 / (wavelength**5) / (np.exp(H*C/(wavelength*K_B*temp)) - 1)
        result = planck_law(wavelength, temp)
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_planck_law_array_input(self):
        """测试数组输入"""
        wavelengths = np.array([400e-9, 500e-9, 600e-9])
        temp = 3000
        results = planck_law(wavelengths, temp)
        self.assertEqual(len(results), 3)
        for i, wl in enumerate(wavelengths):
            expected = 2.0 * H * C**2 / (wl**5) / (np.exp(H*C/(wl*K_B*temp)) - 1)
            self.assertAlmostEqual(results[i], expected, places=10)

class TestVisiblePowerRatio(unittest.TestCase):
    """测试可见光功率比计算"""
    
    def test_visible_power_ratio_at_3000K(self):
        """测试3000K时的可见光功率比"""
        temp = 3000
        result = calculate_visible_power_ratio(temp)
        
        # 手动计算验证
        def intensity(wl):
            return planck_law(wl, temp)
        
        visible_power, _ = quad(intensity, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
        total_power, _ = quad(intensity, 1e-9, 10000e-9)
        expected = visible_power / total_power
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_visible_power_ratio_at_6000K(self):
        """测试6000K时的可见光功率比"""
        temp = 6000
        result = calculate_visible_power_ratio(temp)
        
        # 确保结果在合理范围内
        self.assertGreater(result, 0.1)
        self.assertLess(result, 0.5)

class TestOptimalTemperature(unittest.TestCase):
    """测试最优温度查找"""
    
    def test_optimal_temperature_range(self):
        """测试最优温度在合理范围内"""
        opt_temp, opt_eff = find_optimal_temperature()
        self.assertGreater(opt_temp, 6000)
        self.assertLess(opt_temp, 7000)
        
        # 验证效率确实在最优温度处最大
        eff_plus = calculate_visible_power_ratio(opt_temp + 100)
        eff_minus = calculate_visible_power_ratio(opt_temp - 100)
        self.assertGreater(opt_eff, eff_plus)
        self.assertGreater(opt_eff, eff_minus)

if __name__ == "__main__":
    unittest.main()