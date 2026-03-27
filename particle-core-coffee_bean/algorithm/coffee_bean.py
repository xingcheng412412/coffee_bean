import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib
import os

# 求咖啡豆的面积
def analyze_coffee_area(Ginfos_copy_sort, circle):
    ratio_ = 175.0 / (circle[2] * 2) * 1e3
    actual_areas = []

    # 计算每个颗粒的实际面积（mm^2）
    for pix in Ginfos_copy_sort[:, 1]:
        if pix <= 7000:
            continue
        actual_areas.append(pix * (ratio_ ** 2) / 1000000)  # 转换为mm^2

    # 平均面积
    average_area = np.mean(actual_areas)

    # 打印结果
    print(f"咖啡豆平均面积: {average_area:.2f} mm^2")

    return average_area

# 求咖啡豆的个数
def analyze_coffee_particles(Ginfos_copy_sort):

    particle_count = 0

    # 计算每个颗粒的实际面积（mm^2）
    for pix in Ginfos_copy_sort[:, 1]:
        if pix <= 7000:
            continue
        particle_count += 1

    print(f"咖啡豆个数: {particle_count}")

    return particle_count

# 求咖啡豆的短轴直径真实直径
def analyze_coffee_short_axis(short_axes_pixel, circle):
    """
    计算咖啡豆的短轴真实直径(mm)
    
    参数:
        short_axes_pixel: 短轴直径列表（像素单位）
        circle: ROI圆参数 (x, y, radius)
        
    返回:
        short_axes_real: 短轴真实直径列表(mm)
    """
    if len(short_axes_pixel) == 0:
        print("没有检测到咖啡豆颗粒")
        return [], 0.0
    
    # 计算像素到毫米的转换比例
    pixel_to_mm_ratio = 175.0 / (circle[2] * 2)
    
    # 计算每个颗粒的短轴真实直径（mm）
    short_axes_real = []
    for i, short_axis_pixel in enumerate(short_axes_pixel):
        if short_axis_pixel <= 80:  # 忽略过小的颗粒（80px ≈ 4mm，低于12目下限4.76mm）
            continue

        short_axis_mm = short_axis_pixel * pixel_to_mm_ratio
        short_axes_real.append(short_axis_mm)
        
        # print(f"咖啡豆 {i+1}:")
        # print(f"  短轴直径: {short_axis_mm:.2f} mm")
    
    return short_axes_real

def analyze_coffee_mesh_number(short_axes_real):
    """
    计算咖啡豆的目数(基于短轴直径)
    
    参数:
        short_axes_real: 短轴真实直径列表(mm)
        
    返回:
        mesh_numbers: 目数列表
    """
    mesh_numbers = []

    # 根据短轴直径范围确定目数
    for short_axis_mm in short_axes_real:
        if short_axis_mm > 11.91:
            mesh_numbers.append(31)
        elif short_axis_mm <= 11.91 and short_axis_mm > 11.51:
            mesh_numbers.append(30)
        elif short_axis_mm <= 11.51 and short_axis_mm > 11.11:
            mesh_numbers.append(29)
        elif short_axis_mm <= 11.11 and short_axis_mm > 10.72:
            mesh_numbers.append(28)
        elif short_axis_mm <= 10.72 and short_axis_mm > 10.32:
            mesh_numbers.append(27)
        elif short_axis_mm > 10.32 and short_axis_mm <= 11.11:
            mesh_numbers.append(26)
        elif short_axis_mm <= 10.32 and short_axis_mm > 9.92:
            mesh_numbers.append(25)
        elif short_axis_mm <= 9.92 and short_axis_mm > 9.53:
            mesh_numbers.append(24)
        elif short_axis_mm <= 9.53 and short_axis_mm > 9.13:
            mesh_numbers.append(23)
        elif short_axis_mm <= 9.13 and short_axis_mm > 8.73:
            mesh_numbers.append(22)
        elif short_axis_mm <= 8.73 and short_axis_mm > 8.33:
            mesh_numbers.append(21)
        elif short_axis_mm <= 8.33 and short_axis_mm > 7.93:
            mesh_numbers.append(20)
        elif short_axis_mm <= 7.93 and short_axis_mm > 7.54:
            mesh_numbers.append(19)
        elif short_axis_mm <= 7.54 and short_axis_mm > 7.14:
            mesh_numbers.append(18)
        elif short_axis_mm <= 7.14 and short_axis_mm > 6.75:
            mesh_numbers.append(17)
        elif short_axis_mm <= 6.75 and short_axis_mm > 6.35:
            mesh_numbers.append(16)
        elif short_axis_mm <= 6.35 and short_axis_mm > 5.95:
            mesh_numbers.append(15)
        elif short_axis_mm <= 5.95 and short_axis_mm > 5.55:
            mesh_numbers.append(14)
        elif short_axis_mm <= 5.55 and short_axis_mm > 5.15:
            mesh_numbers.append(13)
        elif short_axis_mm <= 5.15 and short_axis_mm > 4.76:
            mesh_numbers.append(12) 
            
    return mesh_numbers

def analyze_mesh_distribution(mesh_numbers):
    """
    分析咖啡豆目数分布情况
    """
    if not mesh_numbers:
        return None, "无数据"

    mesh_counter = Counter(mesh_numbers)
    max_count = max(mesh_counter.values())
    most_common_meshes = [mesh for mesh, count in mesh_counter.items() if count == max_count]
    most_common_mesh = max(most_common_meshes)

    min_mesh = min(mesh_numbers)
    max_mesh = max(mesh_numbers)
    mesh_range_str = f"{min_mesh}-{max_mesh}"

    print(f"占比最高的目数: {most_common_mesh}")
    print(f"总目数范围: {mesh_range_str}")

    return most_common_mesh, mesh_range_str

def plot_mesh_histogram(mesh_numbers, save_path=None, coffee_kind = 1):
    """
    根据目数列表绘制直方图
    
    参数:
        mesh_numbers: 目数列表
        
    返回:
        无
    """

    # 检查是否有目数数据
    if len(mesh_numbers) == 0:
        print("没有目数数据可绘制直方图")
        return
    
    # 设置中文字体，避免警告
    try:
        # 设置支持中文的字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except:
        pass  # 如果设置失败，继续使用默认字体

    # 统计每个目数的出现次数
    mesh_counts = Counter(mesh_numbers)

    # 设置目数范围（12-30目）
    all_meshes = list(range(12, 31))  # 12到30目
    
    # 获取每个目数的计数，如果没有则为0
    counts = [mesh_counts.get(mesh, 0) for mesh in all_meshes]
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(all_meshes)), counts)
    
    # 设置图表属性
    plt.xlabel('目数')
    plt.ylabel('个数')
    if coffee_kind == 1:
        plt.title('生咖啡豆目数分布')
    else:
        plt.title('烘焙后咖啡豆目数分布')
    plt.xticks(range(len(all_meshes)), all_meshes)
    
    # 保存图片
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"图片已保存到: {save_path}")

# 求咖啡豆的膨胀
def analyze_coffee_expansion(green_coffee_average_area, dark_coffee_average_area):
    # 计算膨胀
    expansion = (dark_coffee_average_area - green_coffee_average_area) / green_coffee_average_area * 100.0
    print(f"膨胀率: {expansion:.2f} %")
    return expansion
    