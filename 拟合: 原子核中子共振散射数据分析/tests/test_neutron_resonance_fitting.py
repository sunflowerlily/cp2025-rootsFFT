import unittest
import numpy as np
from scipy.optimize import curve_fit
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_resonance_fitting_student import breit_wigner, fit_without_errors, fit_with_errors
#from solution.neutron_resonance_fitting_solution import breit_wigner, fit_without_errors, fit_with_errors

class TestBreitWignerFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 实验数据
        cls.energy = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        cls.cross_section = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
        cls.errors = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])
        
        # 参考参数
        cls.Er_ref = 75.0
        cls.Gamma_ref = 50.0
        cls.fr_ref = 10000.0

    def test_breit_wigner_function_points_20(self):
        """测试Breit-Wigner公式实现"""
        # 测试峰值点
        result = breit_wigner(self.Er_ref, self.Er_ref, self.Gamma_ref, self.fr_ref)
        self.assertAlmostEqual(result, 4*self.fr_ref/self.Gamma_ref**2, delta=1e-3)
        
        # 测试远离峰值的点
        result = breit_wigner(self.Er_ref+100, self.Er_ref, self.Gamma_ref, self.fr_ref)
        expected = self.fr_ref / (100**2 + self.Gamma_ref**2/4)
        self.assertAlmostEqual(result, expected, delta=1e-3)

    def test_fit_without_errors_points_30(self):
        """测试不考虑误差的拟合"""
        popt, pcov = fit_without_errors(self.energy, self.cross_section)
        
        # 检查返回参数数量
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3,3))
        
        # 检查参数合理性
        self.assertTrue(50 < popt[0] < 100)  # Er应该在50-100 MeV之间
        self.assertTrue(30 < popt[1] < 100)   # Gamma应该在30-100 MeV之间
        self.assertTrue(100 < popt[2] < 100000)  # 放宽fr的范围到100-100000

    def test_fit_with_errors_points_30(self):
        """测试考虑误差的拟合"""
        popt, pcov = fit_with_errors(self.energy, self.cross_section, self.errors)
        
        # 检查返回参数数量
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3,3))
        
        # 检查参数合理性
        self.assertTrue(50 < popt[0] < 100)  # Er应该在50-100 MeV之间
        self.assertTrue(30 < popt[1] < 100)   # Gamma应该在30-100 MeV之间
        self.assertTrue(100 < popt[2] < 100000)  # 放宽fr的范围到100-100000

    def test_fit_consistency_points_20(self):
        """测试两种拟合方法的一致性"""
        popt1, _ = fit_without_errors(self.energy, self.cross_section)
        popt2, _ = fit_with_errors(self.energy, self.cross_section, self.errors)
        
        # 检查两种方法得到的参数差异在合理范围内
        self.assertAlmostEqual(popt1[0], popt2[0], delta=10)  # Er差异小于10 MeV
        self.assertAlmostEqual(popt1[1], popt2[1], delta=20)   # Gamma差异小于20 MeV
        self.assertAlmostEqual(popt1[2], popt2[2], delta=10000) # fr差异小于10000

if __name__ == "__main__":
    unittest.main()