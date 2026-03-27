from algorithm.GranularRecon import granular_recon
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
def get_coffee_powder_Info(img_file, decimals=3):
    imgarr = np.asarray(bytearray(img_file.read()), dtype='uint8')
    image = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)

    # 解析图片
    GRAs_copy, img_cat_, circle, contours, Ginfos_copy_sort, feature_title, counter_nonconv, counter_skin = granular_recon(image)
    
    bo = Ginfos_copy_sort[:, -1] != 1
    Ginfos_copy_sort = Ginfos_copy_sort[bo, :]

    ratio_ = 175.0 / (circle[2] * 2) * 1e3  # --- 单位 um/pix，修正系数
    coffee_area = Ginfos_copy_sort[:, 1] * (ratio_**2) # --- 折换成圆的面积， um2
    coffee_size = 2 * np.sqrt(coffee_area / np.pi) #粒径_直径
    # 每100μm一个bin，x和y一一对应
    coffee_bins = np.arange(0, 2801, 100)  # [0,100,200,...,2800]

    """ 粒径, x轴刻度（每个bin左边界，与y一一对应） """
    x_data = [int(v) for v in coffee_bins[1:]]  # [100,200,...,2800]

    """ 粒径, y轴 """
    hist_c_1, _ = np.histogram(coffee_size, bins=coffee_bins)
    hist_c_1 = hist_c_1 / np.sum(hist_c_1) * 100
    y_data = [float(np.round(v, decimals)) for v in hist_c_1]

    """ 粒径, y轴-累积分布直方图纵坐标 """
    cums_c_1 = np.cumsum(hist_c_1)
    y_data_accumulate = [float(np.round(v, decimals)) for v in cums_c_1]
    
    most_freq_size = x_data[np.argmax(hist_c_1)]
    
    bo_ = (coffee_size >= 200) & (coffee_size <= 850)
    coffee_size_len = len(coffee_size)
    pass_rate = np.nansum(coffee_area[bo_]) / np.nansum(coffee_area)
    fine_particles = len(coffee_size[(coffee_size > 0) & (coffee_size < 200)])/ coffee_size_len * 100
    medium_particles = len(coffee_size[(coffee_size >= 200) & (coffee_size < 1000)])/ coffee_size_len * 100
    large_particles = len(coffee_size[coffee_size >= 1000])/ coffee_size_len * 100
    digit = [np.mean(coffee_size), #平均粒径
            np.mean(coffee_area), #平均面积
            most_freq_size, #最频粒径
            np.percentile(coffee_size, 50), #数组中大于百分之50的值中的最小值
            np.percentile(coffee_size, 97),#数组中大于百分之97的值中的最小值
            pass_rate,fine_particles,medium_particles,large_particles]

    res_url = "null"

    # 只保留 y_data > 0 的 bin，同步去掉 x_data / y_data_accumulate 对应项
    mask = [y > 0 for y in y_data]
    x_data          = [x  for x,  m in zip(x_data,          mask) if m]
    y_data          = [y  for y,  m in zip(y_data,          mask) if m]
    y_data_accumulate = [ya for ya, m in zip(y_data_accumulate, mask) if m]

    return x_data, y_data, y_data_accumulate, digit, res_url


def draw_powder_distribution(x_data, y_data, y_data_accumulate, save_path=None, title="咖啡粉粒径分布"):
    """
    绘制粒径分布图（柱状图 + 累计折线），双 Y 轴帕累托风格。

    参数
    ----
    x_data           : list[int]    各 bin 左边界（μm）
    y_data           : list[float]  各 bin 占比 %
    y_data_accumulate: list[float]  累计占比 %
    save_path        : str | None   保存路径（None 则只返回不保存）
    title            : str          图表标题

    返回
    ----
    fig : matplotlib Figure 对象
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if not x_data:
        return None

    x_labels = [f"{x}" for x in x_data]

    fig, ax1 = plt.subplots(figsize=(max(10, len(x_data) * 0.6), 5))

    # 柱状图（左轴：占比 %）
    bar_color = "#4C8CBF"
    bars = ax1.bar(x_labels, y_data, color=bar_color, alpha=0.85, zorder=2, label="占比 %")
    ax1.set_xlabel("粒径 (μm)", fontsize=12)
    ax1.set_ylabel("占比 (%)", fontsize=12, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, max(y_data) * 1.35 if y_data else 10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # 柱子顶部标数值（占比 > 1% 才标，避免拥挤）
    for bar, val in zip(bars, y_data):
        if val >= 1.0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=bar_color)

    # 折线图（右轴：累计占比 %）
    ax2 = ax1.twinx()
    line_color = "#E05C2A"
    ax2.plot(x_labels, y_data_accumulate, color=line_color, marker="o", linewidth=2,
             markersize=5, zorder=3, label="累计 %")
    ax2.set_ylabel("累计占比 (%)", fontsize=12, color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # 折线点旁标数值（每隔一个标，避免重叠）
    for i, (xi, ya) in enumerate(zip(x_labels, y_data_accumulate)):
        if i % 2 == 0 or i == len(x_labels) - 1:
            ax2.annotate(f"{ya:.1f}%", xy=(xi, ya),
                         xytext=(4, 4), textcoords="offset points",
                         fontsize=8, color=line_color)

    # 图例合并
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)

    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig

