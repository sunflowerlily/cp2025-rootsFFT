#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯的温度 - 计算白炽灯的最佳工作温度

本模块实现了基于普朗克黑体辐射定律的白炽灯效率计算，
并使用黄金分割法寻找最佳工作温度。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# 物理常数
H = 6.62607015e-34  # 普朗克常数 (J·s)
C = 299792458       # 光速 (m/s)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

# 可见光波长范围 (m)
VISIBLE_LIGHT_MIN = 380e-9  # 380 nm
VISIBLE_LIGHT_MAX = 780e-9  # 780 nm


def planck_law(wavelength, temperature):
    """
    计算普朗克黑体辐射公式
    
    参数:
        wavelength (float or numpy.ndarray): 波长，单位为米
        temperature (float): 温度，单位为开尔文
    
    返回:
        float or numpy.ndarray: 在给定波长和温度下的辐射强度，单位为W/(m^2·m)
    """
    # TODO: 实现普朗克黑体辐射公式 (约5行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用普朗克公式 B(λ,T) = (2hc²/λ⁵) · 1/(e^(hc/(λkT)) - 1)
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return intensity


def plot_blackbody_spectrum(temperatures):
    """
    绘制不同温度下的黑体辐射谱
    
    参数:
        temperatures (list or numpy.ndarray): 温度列表，单位为开尔文
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # TODO: 实现黑体辐射谱的绘制 (约20行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 创建波长数组，计算每个温度下的辐射强度，绘制曲线
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return fig


def calculate_visible_power_ratio(temperature):
    """
    计算给定温度下，可见光波长范围内的辐射功率与总辐射功率的比值
    
    参数:
        temperature (float): 温度，单位为开尔文
    
    返回:
        float: 可见光效率（可见光辐射功率/总辐射功率）
    """
    # TODO: 实现可见光效率的计算 (约20行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用数值积分计算可见光波长范围和全波长范围的辐射功率
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return visible_power_ratio


def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率与温度的关系曲线
    
    参数:
        temp_range (numpy.ndarray): 温度范围，单位为开尔文
    
    返回:
        tuple: (matplotlib.figure.Figure, numpy.ndarray, numpy.ndarray) 图形对象、温度数组、效率数组
    """
    # TODO: 实现效率与温度关系的绘制 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 计算每个温度下的效率，绘制曲线
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return fig, temperatures, efficiencies


def golden_section_search(f, a, b, tol=1e-5):
    """
    使用黄金分割法寻找函数f在区间[a,b]上的最大值
    
    参数:
        f (callable): 目标函数
        a (float): 区间下界
        b (float): 区间上界
        tol (float): 收敛容差
    
    返回:
        tuple: (float, float) 最大值对应的x值和函数值
    """
    # TODO: 实现黄金分割法 (约25行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 实现黄金分割法的迭代过程，寻找函数的最大值
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return x_max, f_max


def find_optimal_temperature():
    """
    寻找使白炽灯效率最大的最佳温度
    
    返回:
        tuple: (float, float) 最佳温度和对应的效率
    """
    # 定义目标函数（效率函数的负值，因为我们要找最大值）
    def objective(temperature):
        return -calculate_visible_power_ratio(temperature)
    
    # 设定搜索区间（单位：K）
    lower_bound = 1000  # 1000K
    upper_bound = 10000  # 10000K
    
    # 使用黄金分割法寻找最佳温度
    optimal_temp, neg_efficiency = golden_section_search(objective, lower_bound, upper_bound)
    optimal_efficiency = -neg_efficiency
    
    return optimal_temp, optimal_efficiency


def main():
    """
    主函数，执行白炽灯最佳温度的计算和可视化
    """
    # 1. 绘制不同温度下的黑体辐射谱
    temperatures = [2000, 3000, 4000, 5000, 6000]  # 单位：K
    fig_spectrum = plot_blackbody_spectrum(temperatures)
    plt.savefig('blackbody_spectrum.png', dpi=300)
    plt.show()
    
    # 2. 绘制效率与温度的关系曲线
    temp_range = np.linspace(1000, 10000, 100)  # 温度范围：1000K到10000K
    fig_efficiency, temps, effs = plot_efficiency_vs_temperature(temp_range)
    plt.savefig('efficiency_vs_temperature.png', dpi=300)
    plt.show()
    
    # 3. 寻找最佳温度
    optimal_temp, optimal_efficiency = find_optimal_temperature()
    print(f"\n最佳温度: {optimal_temp:.1f} K")
    print(f"最大效率: {optimal_efficiency:.4f} ({optimal_efficiency*100:.2f}%)")
    
    # 4. 与实际白炽灯工作温度比较
    actual_temp = 2700  # 典型白炽灯工作温度约为2700K
    actual_efficiency = calculate_visible_power_ratio(actual_temp)
    print(f"\n实际白炽灯工作温度: {actual_temp} K")
    print(f"实际效率: {actual_efficiency:.4f} ({actual_efficiency*100:.2f}%)")
    print(f"效率差异: {(optimal_efficiency - actual_efficiency)*100:.2f}%")
    
    # 5. 在效率曲线上标记最佳点和实际点
    plt.figure(figsize=(10, 6))
    plt.plot(temps, effs, 'b-')
    plt.plot(optimal_temp, optimal_efficiency, 'ro', markersize=8, label=f'最佳温度: {optimal_temp:.1f} K')
    plt.plot(actual_temp, actual_efficiency, 'go', markersize=8, label=f'实际温度: {actual_temp} K')
    plt.xlabel('温度 (K)')
    plt.ylabel('可见光效率')
    plt.title('白炽灯效率与温度的关系')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('optimal_temperature.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()