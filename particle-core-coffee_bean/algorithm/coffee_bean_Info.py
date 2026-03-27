from algorithm.coffee_beans_analyze import *
from algorithm.coffee_bean import *
from utils.upload_client import upload_file
import numpy as np
import cv2
import os
import tempfile
import logging
import time

TASK_ID = "orig"
TOKEN = "T1vxb8f_f8kj-X9BlTteCquWx-U7z0VCTpA18jn1I3HDr6qwOkxn"
UPLOAD_SERVER = os.getenv("UPLOAD_SERVER", "http://127.0.0.1:2090")
MAX_LENGTH = 0

"""
Return image particleSize info

    Parameters
----------
img_file : image url
decimals : retain decimal places
"""
def get_coffee_bean_Info(img_file, decimals=3):
    bean_detect_result = {}
    bean_mesh_range = [0, 0]
    imgarr = np.asarray(bytearray(img_file.read()), dtype='uint8')
    image = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)

    # 解析图片
    circle, Ginfos_copy_sort, short_axes1, valid_contours, img_processed = granular_recon(image)

    # 咖啡豆平均面积
    coffee_average_area = analyze_coffee_area(Ginfos_copy_sort, circle)

    # 咖啡豆个数
    coffee_particle_count = analyze_coffee_particles(Ginfos_copy_sort)

    # 咖啡豆目数（保持与 valid_contours 一一对齐，无法分配目数的置 None）
    pixel_to_mm = 175.0 / (circle[2] * 2)
    mesh_per_contour = []   # 与 valid_contours 等长，None 表示该轮廓无有效目数
    short_axes_real = []    # 仅包含有效短轴，用于统计分布
    for sa_px in short_axes1:
        if sa_px <= 80:
            mesh_per_contour.append(None)
            continue
        sa_mm = sa_px * pixel_to_mm
        meshes = analyze_coffee_mesh_number([sa_mm])
        m = meshes[0] if meshes else 12  # 低于12目下限时仍标记为12目
        mesh_per_contour.append(m)
        short_axes_real.append(sa_mm)

    mesh_number = [m for m in mesh_per_contour if m is not None]
    most_common_mesh, mesh_range = analyze_mesh_distribution(mesh_number)

    bean_mesh_range[0] = min(mesh_number) if mesh_number else 0
    bean_mesh_range[1] = max(mesh_number) if mesh_number else 0

    # 计算目数分布（particle_x/y/y_accumulate），统一转为 Python 原生类型
    if mesh_number:
        mesh_arr = np.array(mesh_number)
        x_data = [float(m) for m in sorted(set(mesh_arr.tolist()))]
        y_counts = [float(np.sum(mesh_arr == m)) for m in x_data]
        total = sum(y_counts)
        y_data = [float(round(cnt / total * 100, 2)) for cnt in y_counts]
        y_accumulate = []
        cumsum = 0.0
        for cnt in y_counts:
            cumsum += cnt
            y_accumulate.append(float(round(cumsum / total * 100, 2)))
    else:
        x_data, y_data, y_accumulate = [], [], []

    bean_detect_result["bean_mesh_range"] = bean_mesh_range
    bean_detect_result["most_freq_mesh"] = most_common_mesh
    bean_detect_result["mesh_freq"] = mesh_number
    bean_detect_result["bean_number"] = coffee_particle_count
    bean_detect_result["bean_details"] = None

    res_url = "null"

    # mesh_per_contour 与 valid_contours 一一对齐（None = 无目数），供绘图用
    return coffee_average_area, res_url, bean_detect_result, x_data, y_data, y_accumulate, img_processed, circle, valid_contours, mesh_per_contour
