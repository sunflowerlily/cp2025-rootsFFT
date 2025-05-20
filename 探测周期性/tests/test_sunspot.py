#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 测试模块
"""

import unittest
import numpy as np
import os
import sys

# 添加父目录到模块搜索路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution.sunspot_solution import (
#from sunspot_student import (
    load_sunspot_data,
    compute_power_spectrum,
    find_main_period
)

class TestSunspotAnalysis(unittest.TestCase):
    """测试太阳黑子数据分析功能"""
    
    @classmethod
    def setUpClass(cls):
        """加载测试数据"""
        cls.years, cls.sunspots = load_sunspot_data("sunspot_data.txt")
    
    def test_data_loading(self):
        """测试数据加载功能"""
        self.assertEqual(len(self.years), len(self.sunspots))
        self.assertGreater(len(self.years), 0)
        self.assertTrue(np.all(self.sunspots >= 0))
    
    def test_power_spectrum_calculation(self):
        """测试功率谱计算"""
        frequencies, power = compute_power_spectrum(self.sunspots)
        
        # 检查返回类型和形状
        self.assertIsInstance(frequencies, np.ndarray)
        self.assertIsInstance(power, np.ndarray)
        self.assertEqual(len(frequencies), len(power))
        
        # 检查功率谱非负
        self.assertTrue(np.all(power >= 0))
    
    def test_main_period_detection(self):
        """测试主周期检测"""
        frequencies, power = compute_power_spectrum(self.sunspots)
        main_period = find_main_period(frequencies, power)
        
        # 检查周期在合理范围内 (10-12年)
        self.assertGreater(main_period, 10*12)  # 10年=120个月
        self.assertLess(main_period, 12*12)    # 12年=144个月
        
        # 检查周期与预期值接近 (约11年)
        self.assertAlmostEqual(main_period/12, 11.0, delta=1.0)

if __name__ == "__main__":
    unittest.main()