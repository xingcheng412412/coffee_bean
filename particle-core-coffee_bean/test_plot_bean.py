import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from algorithm.particle_go import get_partical

# 中文字体
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]
font_path = None
for p in font_candidates:
    if os.path.exists(p):
        font_path = p
        break
if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

IMAGE_PATH = "core/1.jpg"

with open(IMAGE_PATH, "rb") as f:
    success, res = get_partical(f, detect_type=2, filename=IMAGE_PATH)

if not success:
    print("检测失败:", res)
    sys.exit(1)

x = res.particle_x
y = res.particle_y
y_acc = res.particle_y_accumulate

print("particle_x:", x)
print("particle_y:", y)
print("particle_y_accumulate:", y_acc)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.bar(x, y, width=0.4, color='steelblue', alpha=0.7, label='频率 (%)')
ax1.set_xlabel('目数', fontsize=12)
ax1.set_ylabel('频率 (%)', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x)

ax2 = ax1.twinx()
ax2.plot(x, y_acc, color='orangered', marker='o', linewidth=2, label='累计频率 (%)')
ax2.set_ylabel('累计频率 (%)', color='orangered', fontsize=12)
ax2.tick_params(axis='y', labelcolor='orangered')
ax2.set_ylim(0, 110)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title(f'咖啡豆目数分布  (总数: {res.bean_detect_result.get("bean_number", "?")})', fontsize=14)
plt.tight_layout()
plt.savefig("bean_distribution.png", dpi=150)
print("图已保存到 bean_distribution.png")
