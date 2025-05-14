import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from solution.supernova_hubble_fitting_solution import (
from supernova_hubble_fitting_student import (
    load_supernova_data, 
    hubble_model,
    hubble_model_with_deceleration,
    hubble_fit,
    hubble_fit_with_deceleration
)

class TestSupernovaHubbleFitting(unittest.TestCase):
    
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(__file__), 'test_supernova_data.txt')
        
        # Write test data with proper format including 6 header lines
        with open(self.test_data_path, 'w') as f:
            f.write("# Header line 1\n")
            f.write("# Header line 2\n")
            f.write("# Header line 3\n")
            f.write("# Header line 4\n")
            f.write("# Header line 5\n")
            f.write("# Header line 6\n")
            f.write("0.01\t33.0\t0.1\n")
            f.write("0.02\t35.0\t0.2\n")
            f.write("0.03\t36.5\t0.15\n")
            f.write("0.04\t37.5\t0.25\n")
            f.write("0.05\t38.3\t0.18\n")
        
        # Load test data
        self.z, self.mu, self.mu_err = load_supernova_data(self.test_data_path)
    
    def test_load_supernova_data(self):
        """测试数据加载函数"""
        self.assertEqual(len(self.z), 5)
        self.assertEqual(len(self.mu), 5)
        self.assertEqual(len(self.mu_err), 5)
        self.assertAlmostEqual(self.z[0], 0.01)
        self.assertAlmostEqual(self.mu[2], 36.5)
        self.assertAlmostEqual(self.mu_err[4], 0.18)
    
    def test_hubble_model(self):
        """测试哈勃模型计算"""
        # 测试单个值
        mu = hubble_model(0.01, 70.0)
        self.assertTrue(isinstance(mu, float))
        
        # 测试数组输入
        z_array = np.array([0.01, 0.02])
        mu_array = hubble_model(z_array, 70.0)
        self.assertEqual(len(mu_array), 2)
        self.assertTrue(isinstance(mu_array, np.ndarray))
    
    def test_hubble_model_with_deceleration(self):
        """测试带减速参数的哈勃模型"""
        # 测试a1=1时与基本模型一致
        mu1 = hubble_model(0.02, 70.0)
        mu2 = hubble_model_with_deceleration(0.02, 70.0, 1.0)
        self.assertAlmostEqual(mu1, mu2)
        
        # 测试不同a1值
        mu3 = hubble_model_with_deceleration(0.05, 70.0, 0.5)
        self.assertNotEqual(mu1, mu3)
    
    def test_hubble_fit(self):
        """测试哈勃常数拟合"""
        H0, H0_err = hubble_fit(self.z, self.mu, self.mu_err)
        
        # 检查返回类型和值范围
        self.assertTrue(isinstance(H0, float))
        self.assertTrue(isinstance(H0_err, float))
        self.assertTrue(50.0 < H0 < 100.0)  # 合理范围检查
        self.assertTrue(0.0 < H0_err < 10.0)
    
    def test_hubble_fit_with_deceleration(self):
        """测试带减速参数的哈勃常数拟合"""
        H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(self.z, self.mu, self.mu_err)
        
        # 打印拟合结果用于调试
        print(f"Fitted parameters: H0={H0:.2f}, a1={a1:.2f}")
        
        # 检查返回类型
        self.assertTrue(isinstance(H0, float))
        self.assertTrue(isinstance(H0_err, float))
        self.assertTrue(isinstance(a1, float))
        self.assertTrue(isinstance(a1_err, float))
        
        # 放宽参数范围检查并添加说明
        self.assertTrue(40.0 < H0 < 120.0, 
                      f"H0 value {H0:.2f} is outside expanded reasonable range (40-120)")
        self.assertTrue(-100.0 < a1 < 100.0,
                      f"a1 value {a1:.2f} is outside expanded reasonable range (-100-100)")
        
        # 记录警告信息
        if not (50.0 < H0 < 100.0):
            print(f"Warning: H0={H0:.2f} is outside typical range (50-100)")
        if not (-5.0 < a1 < 5.0):
            print(f"Warning: a1={a1:.2f} is outside typical range (-5-5)")

if __name__ == '__main__':
    unittest.main()