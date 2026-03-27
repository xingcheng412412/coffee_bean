# 咖啡豆目数分布图 — 建图文档

---

## 1. 图表参数说明

| 参数 | 含义 | 数据类型 | 图表角色 | 渲染方式 |
|------|------|---------|---------|---------|
| `particle_x` | 咖啡豆目数（去重排序后的目数值） | `float64[]` | 共用 X 轴刻度 | — |
| `particle_y` | 每个目数对应的频率（%） | `float64[]` | 左 Y 轴高度 | 蓝色柱状图 |
| `particle_y_accumulate` | 累积频率（%） | `float64[]` | 右 Y 轴高度 | 橙红色折线 + 圆点标记 |

### 数据来源（`coffee_bean_Info.py`）

```python
mesh_arr = np.array(mesh_number)                           # 每颗豆子的目数列表

x_data   = sorted(set(mesh_arr.tolist()))                  # particle_x：去重排序的目数
y_counts = [np.sum(mesh_arr == m) for m in x_data]        # 每个目数出现次数
total    = sum(y_counts)

particle_y           = [cnt / total * 100 for cnt in y_counts]   # 频率 %
particle_y_accumulate = 逐步累加 y_counts / total * 100           # 累积频率 %
```

---

## 2. 图表轴结构

```
左 Y 轴（蓝色，频率 %）          右 Y 轴（橙红色，累积频率 %）
        |                                        |
  50% ──┤██                               100% ──┤          ●
  40% ──┤██                                90% ──┤        ●
  30% ──┤██  ██                            50% ──┤  ●
  20% ──┤██  ██  ██                         0% ──┼──●──────────
  10% ──┤██  ██  ██  ██                         10  20  30  40
   0% ──┴───────────────── X 轴（目数）
```

| 轴 | 内容 | 颜色 |
|----|------|------|
| X 轴 | 目数值（`particle_x`） | — |
| 左 Y 轴 | 频率 % | 蓝色 `#4C9BE8` |
| 右 Y 轴 | 累积频率 % | 橙红色 `#E84C4C` |

---

## 3. 渲染步骤

### 步骤 1 — 取出数据，对齐长度

```python
x     = np.array(particle_x,            dtype=np.float64)
y     = np.array(particle_y,            dtype=np.float64)
y_acc = np.array(particle_y_accumulate, dtype=np.float64)

# 防止 x / y 长度不一致，截取最短
n = min(len(x), len(y), len(y_acc))
x, y, y_acc = x[:n], y[:n], y_acc[:n]
```

### 步骤 2 — 建立画布，创建左 Y 轴（柱状图）

```python
fig, ax1 = plt.subplots(figsize=(10, 5))

# 柱宽 = 相邻目数最小间隔的 60%
bar_width = float(np.min(np.diff(np.sort(x)))) * 0.6

ax1.bar(x, y, width=bar_width, color="#4C9BE8", alpha=0.75, label="频率 (%)")
ax1.set_xlabel("目数 (Mesh Number)")
ax1.set_ylabel("频率 (%)", color="#4C9BE8")
```

### 步骤 3 — 共享 X 轴，叠加右 Y 轴（折线图）

```python
ax2 = ax1.twinx()   # 共享 X 轴，独立右 Y 轴

ax2.plot(x, y_acc, color="#E84C4C", linewidth=2,
         marker="o", markersize=4, label="累积频率 (%)")
ax2.set_ylabel("累积频率 (%)", color="#E84C4C")
ax2.set_ylim(0, 110)
```

### 步骤 4 — 合并两轴图例，保存

```python
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
```

---

## 4. 数据到图的对应关系

```
particle_x            = [ 10,    20,    30  ]
                          ↓      ↓      ↓
                         柱位   柱位   柱位   ← ax1.bar()  左Y轴（蓝色柱）
                         折点   折点   折点   ← ax2.plot() 右Y轴（橙红折线）

particle_y            = [33.33, 50.0,  16.67]  → 每根柱的高度（频率 %）
particle_y_accumulate = [33.33, 83.33, 100.0]  → 每个折点的高度（累积频率 %）
```

**示例数据可视化：**

```
左Y轴(频率%)            右Y轴(累积%)
   50 │    █              100 │          ●
   40 │    █               83 │     ●
   33 │█   █                  │
   20 │█   █   █           33 │●
    0 └────────────          └────────────
       10  20  30              10  20  30
```

---

## 5. 完整调用示例

```python
from tests.plot_particle_distribution import plot_particle_distribution

# 来自 get_partical(detect_type=1 或 2) 的结果
particle_x            = result.particle_x            # float64[]
particle_y            = result.particle_y            # float64[]
particle_y_accumulate = result.particle_y_accumulate # float64[]

plot_particle_distribution(
    particle_x,
    particle_y,
    particle_y_accumulate,
    save_path="output/bean_mesh_distribution.png",
    title="咖啡豆目数分布",
)
```

> **detect_type 对照**
> | detect_type | 检测对象 | particle_x 含义 |
> |-------------|---------|----------------|
> | `0` | 咖啡粉粒径 | 粒径区间中值（μm），共24个点，0~2500μm |
> | `1` / `2` | 咖啡豆目数 | 去重排序后的目数值（无固定范围） |
