#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯的温度 - 计算白炽灯的最佳工作温度（参考答案）

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
    # 普朗克黑体辐射公式: B(λ,T) = (2hc²/λ⁵) · 1/(e^(hc/(λkT)) - 1)
    # 计算分子部分: 2hc²/λ⁵
    numerator = 2.0 * H * C**2 / (wavelength**5)
    
    # 计算指数部分: e^(hc/(λkT))
    exponent = np.exp(H * C / (wavelength * K_B * temperature))
    
    # 计算完整公式
    intensity = numerator / (exponent - 1.0)
    
    return intensity


def plot_blackbody_spectrum(temperatures):
    """
    绘制不同温度下的黑体辐射谱
    
    参数:
        temperatures (list or numpy.ndarray): 温度列表，单位为开尔文
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # 创建波长数组（从100nm到3000nm，包含可见光范围）
    wavelengths = np.linspace(100e-9, 3000e-9, 1000)  # 单位：米
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 为每个温度绘制辐射谱
    for temp in temperatures:
        # 计算辐射强度
        intensities = planck_law(wavelengths, temp)
        
        # 绘制曲线
        ax.plot(wavelengths * 1e9, intensities, label=f'{temp} K')
    
    # 标记可见光范围
    ax.axvspan(VISIBLE_LIGHT_MIN * 1e9, VISIBLE_LIGHT_MAX * 1e9, 
               alpha=0.2, color='yellow', label='可见光范围')
    
    # 添加标题和标签
    ax.set_title('不同温度下的黑体辐射谱')
    ax.set_xlabel('波长 (nm)')
    ax.set_ylabel('辐射强度 (W/(m²·m))')
    
    # 使用对数刻度以更好地显示不同波长范围的变化
    ax.set_yscale('log')
    
    # 添加图例
    ax.legend()
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    fig.tight_layout()
    
    return fig


def calculate_visible_power_ratio(temperature):
    """
    计算给定温度下，可见光波长范围内的辐射功率与总辐射功率的比值
    
    参数:
        temperature (float): 温度，单位为开尔文
    
    返回:
        float: 可见光效率（可见光辐射功率/总辐射功率）
    """
    # 定义辐射强度函数（用于积分）
    def intensity_function(wavelength):
        return planck_law(wavelength, temperature)
    
    # 计算可见光波长范围内的辐射功率
    visible_power, _ = integrate.quad(intensity_function, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
    
    # 计算全波长范围内的总辐射功率
    # 注意：理论上，积分范围应该是[0, ∞)，但实际上我们可以使用一个足够大的上限
    # 对于大多数温度，波长超过10000nm的辐射能量可以忽略不计
    total_power, _ = integrate.quad(intensity_function, 1e-9, 10000e-9)
    
    # 计算可见光效率
    visible_power_ratio = visible_power / total_power
    
    return visible_power_ratio


def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率与温度的关系曲线
    
    参数:
        temp_range (numpy.ndarray): 温度范围，单位为开尔文
    
    返回:
        tuple: (matplotlib.figure.Figure, numpy.ndarray, numpy.ndarray) 图形对象、温度数组、效率数组
    """
    # 计算每个温度下的效率
    efficiencies = np.array([calculate_visible_power_ratio(temp) for temp in temp_range])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制效率曲线
    ax.plot(temp_range, efficiencies, 'b-')
    
    # 找出最大效率点
    max_idx = np.argmax(efficiencies)
    max_temp = temp_range[max_idx]
    max_efficiency = efficiencies[max_idx]
    
    # 在图上标记最大效率点
    ax.plot(max_temp, max_efficiency, 'ro', markersize=8)
    ax.text(max_temp, max_efficiency * 0.95, 
            f'最大效率: {max_efficiency:.4f}\n温度: {max_temp:.1f} K', 
            ha='center')
    
    # 添加标题和标签
    ax.set_title('白炽灯效率与温度的关系')
    ax.set_xlabel('温度 (K)')
    ax.set_ylabel('可见光效率')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    fig.tight_layout()
    
    return fig, temp_range, efficiencies


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
    # 黄金比例
    golden_ratio = (np.sqrt(5) - 1) / 2  # 约0.618
    
    # 计算初始测试点
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    
    # 计算函数值
    fc = f(c)
    fd = f(d)
    
    # 迭代直到区间足够小
    while abs(b - a) > tol:
        if fc < fd:  # 因为我们传入的是负的效率函数，所以这里是找最小值
            # 区间缩小为[a,d]
            b = d
            d = c
            fd = fc
            c = b - golden_ratio * (b - a)
            fc = f(c)
        else:
            # 区间缩小为[c,b]
            a = c
            c = d
            fc = fd
            d = a + golden_ratio * (b - a)
            fd = f(d)
    
    # 取区间中点作为最终结果
    x_max = (a + b) / 2
    f_max = f(x_max)
    
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
    
    # 6. 分析为什么实际白炽灯的工作温度低于理论最佳温度
    print("\n为什么实际白炽灯的工作温度低于理论最佳温度？")
    print("1. 灯丝材料（通常是钨）的熔点限制了最高工作温度（钨的熔点约为3695K）")
    print("2. 在更高温度下，灯丝的蒸发速率增加，导致灯泡寿命显著缩短")
    print("3. 更高的温度需要更多的电能输入，增加了能耗和成本")
    print("4. 实际白炽灯不是完美的黑体辐射体，其发射率随波长变化")
    print("5. 商业考量：制造商需要在效率、寿命和成本之间取得平衡")


if __name__ == "__main__":
    main()