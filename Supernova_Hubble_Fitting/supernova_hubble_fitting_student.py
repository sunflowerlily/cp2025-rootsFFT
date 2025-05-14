import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_supernova_data(file_path):
    """
    从文件中加载超新星数据
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: 包含以下元素的元组
            - z (numpy.ndarray): 红移数据
            - mu (numpy.ndarray): 距离模数数据
            - mu_err (numpy.ndarray): 距离模数误差
    """
    #TODO: 实现数据加载功能 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return z, mu, mu_err


def hubble_model(z, H0):
    """
    哈勃模型：距离模数与红移的关系
    
    参数:
        z (float or numpy.ndarray): 红移
        H0 (float): 哈勃常数 (km/s/Mpc)
        
    返回:
        float or numpy.ndarray: 距离模数
    """
    #TODO: 实现哈勃模型计算 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return mu


def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的哈勃模型
    
    参数:
        z (float or numpy.ndarray): 红移
        H0 (float): 哈勃常数 (km/s/Mpc)
        a1 (float): 拟合参数，对应于减速参数q0
        
    返回:
        float or numpy.ndarray: 距离模数
    """
    #TODO: 实现带减速参数的哈勃模型 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return mu


def hubble_fit(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        
    返回:
        tuple: 包含以下元素的元组
            - H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
            - H0_err (float): 哈勃常数的误差
    """
    #TODO: 实现哈勃常数拟合 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return H0, H0_err


def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数和减速参数
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        
    返回:
        tuple: 包含以下元素的元组
            - H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
            - H0_err (float): 哈勃常数的误差
            - a1 (float): 拟合得到的a1参数
            - a1_err (float): a1参数的误差
    """
    #TODO: 实现带减速参数的哈勃常数拟合 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return H0, H0_err, a1, a1_err


def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制哈勃图（距离模数vs红移）
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    #TODO: 实现哈勃图绘制 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return plt.gcf()


def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
        a1 (float): 拟合得到的a1参数
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    #TODO: 实现带减速参数的哈勃图绘制 (大约 [X] 行代码)
    #[STUDENT_CODE_HERE]
    #raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    #return plt.gcf()


if __name__ == "__main__":
    # 数据文件路径
    data_file = "data/supernova_data.txt"
    
    # 加载数据
    z, mu, mu_err = load_supernova_data(data_file)
    
    # 拟合哈勃常数
    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    
    # 绘制哈勃图
    fig = plot_hubble_diagram(z, mu, mu_err, H0)
    plt.show()
    
    # 可选：拟合包含减速参数的模型
    H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"拟合得到的a1参数: a1 = {a1:.2f} ± {a1_err:.2f}")
    
    # 绘制包含减速参数的哈勃图
    fig = plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1)
    plt.show()