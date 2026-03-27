# 咖啡粉粒径分布图 — 建图文档

---

## 1. 图表参数说明

| 参数 | 含义 | 数据类型 | 图表角色 | 渲染方式 |
|------|------|---------|---------|---------|
| `particle_x` | 粒径区间中值，单位 **μm** | `float64[]` | 共用 X 轴刻度 | — |
| `particle_y` | 每区间频率（%） | `float64[]` | 左 Y 轴高度 | 蓝色柱状图 |
| `particle_y_accumulate` | 累积频率（%） | `float64[]` | 右 Y 轴高度 | 红色折线 + 圆点标记 |

---

## 2. 数据来源（`coffee_powder_Info.py`）

### 2.1 像素 → 粒径（μm）转换

```python
ratio_ = 175.0 / (circle[2] * 2) * 1e3   # 单位换算系数：μm/pix
                                           # circle[2] 是图像中标定圆半径（pix）
                                           # 175.0 mm = 标定圆直径实物尺寸

coffee_area = Ginfos_copy_sort[:, 1] * (ratio_**2)  # 每颗颗粒面积（μm²）
coffee_size = 2 * np.sqrt(coffee_area / np.pi)       # 等效直径（μm）= 粒径
```

> 粒径计算原理：将颗粒面积等效为圆，直径 = 2 × √(面积/π)

---

### 2.2 particle_x 的生成（粒径区间中值）

```python
coffee_bins  = np.linspace(0, 2501, 25)          # 生成 25 个等距边界点
                                                  # → [0, 104.2, 208.4, ..., 2500]
bin_edges_2  = (coffee_bins[:-1] + coffee_bins[1:]) / 2  # 相邻两边界取中值
                                                  # → 24 个区间中值
particle_x   = [52.1, 156.3, 260.4, ..., 2447.9] # 共 24 个点，单位 μm
```

| 区间编号 | 区间范围 (μm) | 区间中值 (μm)（particle_x） |
|---------|-------------|--------------------------|
| 0 | 0 ~ 104.2 | 52.1 |
| 1 | 104.2 ~ 208.4 | 156.3 |
| 2 | 208.4 ~ 312.5 | 260.4 |
| … | … | … |
| 23 | 2395.8 ~ 2500 | 2447.9 |

---

### 2.3 particle_y 的生成（频率 %）

```python
hist_c_1, _ = np.histogram(coffee_size, bins=25, range=(0, 2500))
# → 统计每个区间内颗粒数量，共 25 个计数（与 particle_x 的 24 个点存在长度差异，绘图时截齐）

hist_c_1 = hist_c_1 / np.sum(hist_c_1) * 100    # 归一化为百分比
particle_y = [round(v, 3) for v in hist_c_1]     # 每区间频率 %
```

**含义：** 落在该粒径区间内的颗粒数 ÷ 总颗粒数 × 100%

---

### 2.4 particle_y_accumulate 的生成（累积频率 %）

```python
cums_c_1 = np.cumsum(hist_c_1)                   # 逐步累加频率
particle_y_accumulate = [round(v, 3) for v in cums_c_1]
```

**含义：** 粒径 ≤ 该区间上边界的颗粒占总颗粒的百分比，最终收敛至 100%

---

## 3. 图表轴结构

```
左 Y 轴（蓝色，频率 %）               右 Y 轴（红色，累积频率 %）
        │                                            │
  30% ──┤    ████                           100% ──┤              ●─●
  25% ──┤    ████                            80% ──┤         ●
  20% ──┤    ████ ████                       50% ──┤    ●
  10% ──┤████████ ████ ████                  20% ──┤●
   0% ──┴──────────────────── X 轴（μm）      0% ──┘
        52  156  260  365 ...                    52  156  260  365
```

| 轴 | 内容 | 颜色 | 范围 |
|----|------|------|------|
| X 轴 | 粒径区间中值（μm） | — | 0 ~ 2500 μm |
| 左 Y 轴 | 频率 %（柱状图） | 蓝色 `#4C9BE8` | 0 ~ max(y) × 1.35 |
| 右 Y 轴 | 累积频率 %（折线图） | 红色 `#E84C4C` | 0 ~ 115% |

---

## 4. 渲染步骤

### 步骤 1 — 取出数据，对齐长度

```python
x     = np.array(particle_x,            dtype=np.float64)  # 24 个粒径中值
y     = np.array(particle_y,            dtype=np.float64)  # 25 个频率（截取前24）
y_acc = np.array(particle_y_accumulate, dtype=np.float64)  # 25 个累积（截取前24）

# x(24) 与 y(25) 长度不一致：histogram bins=25 产生25个计数，但区间中值只有24个
n = min(len(x), len(y), len(y_acc))
x, y, y_acc = x[:n], y[:n], y_acc[:n]   # 统一截为 24 个点
```

> **长度差异原因：**
> `np.linspace(0,2501,25)` 生成25个边界 → 24个区间中值（particle_x 共24点）
> `np.histogram(..., bins=25)` 生成25个计数（particle_y 共25点）
> 绘图前取 `min(len)` 截齐即可。

---

### 步骤 2 — 建立画布，创建左 Y 轴（柱状图）

```python
fig, ax1 = plt.subplots(figsize=(12, 5))

# 柱宽 = 相邻区间间隔（约 104μm）× 0.85，留出间隙
bar_width = float(x[1] - x[0]) * 0.85

ax1.bar(x, y, width=bar_width, color="#4C9BE8", alpha=0.75, label="频率 (%)")
ax1.set_xlabel("粒径 (μm)")
ax1.set_ylabel("频率 (%)", color="#2060A0")
ax1.set_xlim(x[0] - bar_width, x[-1] + bar_width)
ax1.set_ylim(0, max(y) * 1.35)
```

---

### 步骤 3 — 共享 X 轴，叠加右 Y 轴（折线图）

```python
ax2 = ax1.twinx()   # 共享 X 轴，独立右 Y 轴

ax2.plot(x, y_acc, color="#E84C4C", linewidth=2,
         marker="o", markersize=4, label="累积频率 (%)")
ax2.set_ylabel("累积频率 (%)", color="#C02020")
ax2.set_ylim(0, 115)

# D50 / D97 参考线
ax2.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)  # D50
ax2.axhline(97, color="gray", linestyle=":",  linewidth=1, alpha=0.7)  # D97
```

---

### 步骤 4 — 合并图例，保存图片

```python
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
```

---

## 5. 数据到图的对应关系

```
particle_x            = [ 52.1,  156.3,  260.4,  364.6, ... ]  ← 区间中值（μm）
                           ↓       ↓       ↓       ↓
                          柱位    柱位    柱位    柱位    ← ax1.bar()  左Y轴（蓝色柱）
                          折点    折点    折点    折点    ← ax2.plot() 右Y轴（红色折线）

particle_y            = [  5.2,   28.3,   31.1,   18.4, ... ]  → 每根柱的高度（频率 %）
particle_y_accumulate = [  5.2,   33.5,   64.6,   83.0, ... ]  → 每个折点的高度（累积 %）
```

---

## 6. 附加统计指标（`digit` 数组）

由 `get_coffee_powder_Info` 同步返回，对应 `MParticleResult` 各字段：

| 索引 | 字段名 | 计算方式 | 单位 |
|------|--------|---------|------|
| `digit[0]` | `ave_particle_size` | `np.mean(coffee_size)` | μm |
| `digit[1]` | `ave_area` | `np.mean(coffee_area)` | μm² |
| `digit[2]` | `most_frequent_particle_size` | 频率最高区间的中值 | μm |
| `digit[3]` | `mid_particle_size` (D50) | `np.percentile(coffee_size, 50)` | μm |
| `digit[4]` | `high_particle_size` (D97) | `np.percentile(coffee_size, 97)` | μm |
| `digit[5]` | `pass_rate` | 200~850μm 区间面积占比 | — |
| `digit[6]` | `fine_particles` | 0~200μm 颗粒数占比 | % |
| `digit[7]` | `medium_particles` | 200~1000μm 颗粒数占比 | % |
| `digit[8]` | `large_particles` | >1000μm 颗粒数占比 | % |

---

## 7. 完整调用示例

```python
import sys
sys.path.insert(0, ".")

from algorithm.particle_go import get_partical
from tests.plot_particle_distribution import plot_particle_distribution

with open("core/222.jpg", "rb") as f:
    success, result = get_partical(f, detect_type=0, filename="222.jpg")

print(f"平均粒径: {result.ave_particle_size:.2f} μm")
print(f"D50     : {result.mid_particle_size:.2f} μm")
print(f"D97     : {result.high_particle_size:.2f} μm")

plot_particle_distribution(
    result.particle_x,
    result.particle_y,
    result.particle_y_accumulate,
    save_path="output/particle_distribution.png",
    title="咖啡粉粒径分布",
)
```

> **detect_type 对照**
> | detect_type | 检测对象 | particle_x 含义 | 点数 |
> |-------------|---------|----------------|------|
> | `0` | 咖啡粉粒径 | 粒径区间中值（μm），0~2500μm | 24 个等距点 |
> | `1` / `2` | 咖啡豆目数 | 去重排序后的实际目数值 | 不固定 |
