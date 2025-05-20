#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算

本模块实现了一维方势阱中粒子能级的计算方法。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # 电子伏转换为焦耳的系数


def calculate_y_values(E_values, V, w, m):
    """
    计算方势阱能级方程中的三个函数值
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
    
    返回:
        tuple: 包含三个numpy数组 (y1, y2, y3)，分别对应三个函数在给定能量值下的函数值
    """
    # TODO: 实现计算y1, y2, y3的代码 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 注意单位转换和避免数值计算中的溢出或下溢
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    """
    绘制能级方程的三个函数曲线
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        y1 (numpy.ndarray): 函数y1的值
        y2 (numpy.ndarray): 函数y2的值
        y3 (numpy.ndarray): 函数y3的值
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # TODO: 实现绘制三个函数曲线的代码 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用不同颜色和线型，添加适当的标签、图例和标题
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return fig


def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    """
    使用二分法求解方势阱中的第n个能级
    
    参数:
        n (int): 能级序号 (0表示基态，1表示第一激发态，以此类推)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
        precision (float): 求解精度 (eV)
        E_min (float): 能量搜索下限 (eV)
        E_max (float): 能量搜索上限 (eV)，默认为V
    
    返回:
        float: 第n个能级的能量值 (eV)
    """
    # TODO: 实现二分法求解能级的代码 (约25行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 需要考虑能级的奇偶性，偶数能级使用偶宇称方程，奇数能级使用奇宇称方程
    
    raise NotImplementedError("请在 {} 中实现此函数。".format(__file__))
    
    return energy_level


def main():
    """
    主函数，执行方势阱能级的计算和可视化
    """
    # 参数设置
    V = 20.0  # 势阱高度 (eV)
    w = 1e-9  # 势阱宽度 (m)
    m = ELECTRON_MASS  # 粒子质量 (kg)
    
    # 1. 计算并绘制函数曲线
    E_values = np.linspace(0.001, 19.999, 1000)  # 能量范围 (eV)
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()
    
    # 2. 使用二分法计算前6个能级
    energy_levels = []
    for n in range(6):
        energy = find_energy_level_bisection(n, V, w, m)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")
    
    # 与参考值比较
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    print("\n参考能级值:")
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")


if __name__ == "__main__":
    main()