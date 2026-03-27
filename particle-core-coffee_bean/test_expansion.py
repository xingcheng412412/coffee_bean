"""
咖啡豆膨胀率测试
用法：python test_expansion.py <生豆图片路径> <烘焙后图片路径>

膨胀率公式：(烘焙后平均面积 - 生豆平均面积) / 生豆平均面积 × 100%
ave_particle_size 在咖啡豆模式下存储的是平均面积（单位：mm²）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from algorithm.particle_go import get_partical
from algorithm.coffee_bean import analyze_coffee_expansion


def get_ave_area(image_path: str) -> float:
    """调用 get_partical 获取咖啡豆平均面积（ave_particle_size，单位：mm²）"""
    with open(image_path, "rb") as f:
        success, res = get_partical(f, detect_type=2, filename=image_path)
    if not success:
        raise RuntimeError(f"检测失败：{image_path}，原因：{res}")
    return res.ave_particle_size


def main():
    # 支持命令行传参，默认使用 core/ 下的示例图片
    if len(sys.argv) == 3:
        green_path  = sys.argv[1]  # 生豆图片
        roasted_path = sys.argv[2]  # 烘焙后图片
    else:
        green_path   = "core/1.jpg"
        roasted_path = "core/2.jpg"
        print(f"未指定图片路径，使用默认图片：{green_path} / {roasted_path}\n")

    print(f"[生豆]    检测图片：{green_path}")
    green_area = get_ave_area(green_path)
    print(f"[生豆]    ave_particle_size（平均面积）= {green_area:.4f} mm²\n")

    print(f"[烘焙后]  检测图片：{roasted_path}")
    roasted_area = get_ave_area(roasted_path)
    print(f"[烘焙后]  ave_particle_size（平均面积）= {roasted_area:.4f} mm²\n")

    # 调用已有膨胀率函数：(烘后 - 生豆) / 生豆 * 100
    expansion = analyze_coffee_expansion(green_area, roasted_area)
    print(f"膨胀率 = ({roasted_area:.4f} - {green_area:.4f}) / {green_area:.4f} × 100 = {expansion:.2f} %")


if __name__ == "__main__":
    main()
