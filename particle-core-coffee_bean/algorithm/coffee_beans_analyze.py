#%% 导入库
import numpy as np
import cv2

#%% 配置常量
IMAGE_SIZE = 3500  # 1750 * 2
COLOR_THRESHOLD = 210

#%% 图像预处理和ROI检测
def preprocess_img(img_cv):
    """预处理图像：灰度化、检测圆形ROI、背景去除和标准化裁切"""
    # 转为灰度图
    img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # 缩放图像
    size_ = min(img.shape)
    ratio_ = IMAGE_SIZE / size_
    shape_ = (int(img.shape[1]*ratio_), int(img.shape[0]*ratio_))
    img = cv2.resize(img, shape_)

    # 缩小图像用于圆盘检测
    tmp = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
    _, binary = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找最大白色连通区域，拟合最小外接圆
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到圆形 ROI，请检查图像或调整参数")
    largest = max(contours, key=cv2.contourArea)
    (cx_small, cy_small), r_small = cv2.minEnclosingCircle(largest)
    circle = np.array([cx_small * 10, cy_small * 10, r_small * 10])

    # 裁切边界 clamp 到图像范围内
    x0 = int(max(0, circle[0] - circle[2]))
    x1 = int(min(img.shape[1], circle[0] + circle[2]))
    y0 = int(max(0, circle[1] - circle[2]))
    y1 = int(min(img.shape[0], circle[1] + circle[2]))

    # 创建圆形掩膜
    mx, my = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    mask = np.sqrt((mx - circle[0])**2 + (my - circle[1])**2) < circle[2]
    vtmp1 = img[mask]
    img_ = img.copy()
    img_[~mask] = np.percentile(vtmp1, 5)

    # 高斯滤波和局部归一化（缩4x计算后放大，速度提升约70x）
    _scale = 4
    _small = cv2.resize(img_, (img_.shape[1]//_scale, img_.shape[0]//_scale))
    _Gsmall = cv2.GaussianBlur(_small.astype(np.float32) / 255.0, (0, 0), 160 // _scale)
    Gimg = cv2.resize(_Gsmall, (img_.shape[1], img_.shape[0]))

    img_new = img_.astype(np.float64) / 255.0 / (Gimg + 1e-8)
    img_new = np.clip(img_new, 0, 1) * 255
    img_new = img_new.astype(np.uint8)

    # 裁切ROI区域
    img_cat = img_new[y0:y1, x0:x1]

    # 创建圆形掩膜并处理边缘（基于实际裁切尺寸）
    crop_h, crop_w = img_cat.shape[:2]
    cx_crop = int(circle[0]) - x0
    cy_crop = int(circle[1]) - y0
    r_crop  = int(circle[2])
    mx, my = np.meshgrid(np.arange(crop_w), np.arange(crop_h))
    mask = np.sqrt((mx - cx_crop)**2 + (my - cy_crop)**2) > (r_crop - 20)
    img_cat_ = img_cat.copy()
    img_cat_[mask] = 255

    # 标准化尺寸
    circle = (circle * IMAGE_SIZE / img_cat_.shape[0]).astype(np.int32)
    img_cat_ = cv2.resize(img_cat_, (IMAGE_SIZE, IMAGE_SIZE))

    return img_cat_, circle

#%% 轮廓平滑
def smooth_granular(gra):
    """平滑轮廓点"""
    if gra.shape[0] < 3:
        return gra.squeeze()

    gra = gra.squeeze().astype(float)
    n = gra.shape[0]
    n2 = n // 2
    gra_ext = np.vstack((gra, gra[1:, :]))

    # 应用均值滤波
    for _ in range(1):
        for i in range(gra_ext.shape[1]):
            vec = gra_ext[:, i]
            for j in range(1, n2+n-1):
                vec[j] = vec[j-1] + (vec[j+1] - vec[j-1]) / 3

    return gra_ext[n2+1:n2+n+1, :]

#%% 几何特征计算
def cal_area(points):
    """计算多边形面积"""
    if len(points) < 3:
        return 0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def cal_perimeter(points):
    """计算多边形周长"""
    if points.shape[0] < 2:
        return 0
    diffs = np.diff(points, axis=0, prepend=points[-1:])
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def calculate_short_axis(points):
    """计算咖啡豆的短轴直径（像素单位）"""
    if len(points) < 5:
        rect = cv2.minAreaRect(points.astype(np.float32))
        return min(rect[1])
    ellipse = cv2.fitEllipse(points.astype(np.float32))
    return min(ellipse[1])


def calculate_short_axis_robust(points):
    """
    两次判断短轴（防止弧形/薄片轮廓导致 fitEllipse 虚高）：
    第一次：fitEllipse 快速估算。
    第二次：若结果明显偏大（> 距离变换最大内切圆直径 * 1.5），
            则以距离变换值为准——真正粗壮的豆子内切圆半径接近短轴/2；
            薄弧形轮廓 dist.max 极小，说明 fitEllipse 有误。
    """
    pts = points.reshape(-1, 2)
    sa = calculate_short_axis(pts)

    # 用距离变换计算最大内切圆直径
    pad = 3
    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = int(pts[:, 0].max()) + pad + 1
    y1 = int(pts[:, 1].max()) + pad + 1
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return sa
    mask = np.zeros((h, w), dtype=np.uint8)
    lp = (pts - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(mask, [lp], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inscribed_diam = dist.max() * 2

    # 若 fitEllipse 短轴 > 内切圆直径 * 1.5，说明估算有误，取内切圆直径
    if inscribed_diam > 0 and sa > inscribed_diam * 1.5:
        return inscribed_diam
    return sa

def get_granular_info(points):
    """计算颗粒几何特征"""
    center = np.mean(points, axis=0)
    points_float = points.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(points_float).reshape(-1, 2)

    area = cal_area(points)
    convex_area = cal_area(hull)
    perimeter = cal_perimeter(points)
    perimeter_conv = cal_perimeter(hull)

    if area < 1 or convex_area < 1:
        area = convex_area = perimeter = perimeter_conv = points.shape[0]

    shape_ratio = area / convex_area if convex_area > 0 else 0
    short_axis_pixel = calculate_short_axis(points)

    return [
        points.shape[0],   # 像素周长
        area,              # 实际面积
        convex_area,       # 凸包面积
        shape_ratio,       # 形状系数
        perimeter,         # 周长
        perimeter_conv,    # 凸包周长
        center[0],         # 中心X
        center[1],         # 中心Y
        short_axis_pixel   # 短轴直径（像素）
    ]

#%% 粘连颗粒拆分
def _min_neck_ratio(points):
    """
    计算轮廓的最小颈部宽度比 = min(欧氏距离 / 轮廓弧长距离)。
    两豆相切时：欧氏距离≈0，弧长≈N/2，比值≈0（极小）。
    单豆凹陷时：两点均在同一侧，弧长相对较短，比值较大。
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    Dmat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(Dmat, 1)
    idx_diff = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n))
    D2 = np.where(idx_diff > n // 2, n - idx_diff, idx_diff)
    D2[D2 == 0] = 1
    conv_ratio = Dmat / D2
    mask = (idx_diff < 3) | (idx_diff > n - 3)
    conv_ratio[mask] = np.inf
    return float(np.nanmin(conv_ratio))


def is_split_granular(points):
    """判断是否需要拆分粘连颗粒"""
    if points.shape[0] < 10:
        return False
    area = cal_area(points)
    if area < 4:
        return False
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32))
    convex_area = cal_area(hull.reshape(-1, 2))
    shape_ratio = area / convex_area if convex_area > 0 else 1

    # 情形1：明显粘连（形状系数低）且存在真正的颈部收缩
    # 注意：单颗天然凹形豆 sr 也可能 < 0.85，但无真正颈部（neck 较大），不应拆分
    neck = _min_neck_ratio(points)
    if shape_ratio < 0.85 and neck < 0.20:
        return True

    # 情形2：接触/相切——形状系数接近1，但轮廓上存在颈部收缩
    # 颈宽比 < 0.15：覆盖轻微到中等程度的粘连（面积 > 15000）
    if area > 15000 and neck < 0.15:
        return True
    # 颈宽比 < 0.20：大簇（面积 > 35000）允许更宽的颈部
    if area > 35000 and neck < 0.20:
        return True

    # 情形3：两豆紧密重叠——形状系数/颈宽比均接近单豆，但椭圆长短轴比过大
    # 单颗咖啡豆长短轴比通常 ≤ 1.8；两豆叠放时合并椭圆比值 > 1.95
    if area > 12000:
        try:
            ell = cv2.fitEllipse(points.astype(np.float32))
            axes = ell[1]  # (width, height) 即两轴长度
            aspect = max(axes) / min(axes) if min(axes) > 0 else 1.0
            if aspect > 1.95:
                return True
        except Exception:
            pass

    return False

def split_granular(points, split_idx):
    """拆分颗粒"""
    idx1 = np.arange(split_idx[0], split_idx[1] + 1)
    idx2 = np.concatenate([
        np.arange(0, split_idx[0] + 1),
        np.arange(split_idx[1], points.shape[0])
    ])
    part1 = points[idx1]
    part2 = points[idx2]
    if part1.shape[0] > 4:
        for i in range(-1, 3):
            idx = (split_idx[0] + i) % points.shape[0]
            neighbors = [(idx - 1) % points.shape[0], idx, (idx + 1) % points.shape[0]]
            part1[max(0, i), :] = np.mean(points[neighbors, :], axis=0)
    return part1, part2

def _shape_ratio(points):
    """计算轮廓的形状系数（面积/凸包面积）"""
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32))
    convex_area = cal_area(hull.reshape(-1, 2))
    if convex_area < 1:
        return 0.0
    return cal_area(points) / convex_area


def _has_two_peaks(points):
    """距离变换检测，确认轮廓内部是否有2个豆的峰（防止单颗豆的平切假颈触发误拆）"""
    pts = points.reshape(-1, 2)
    x0 = max(0, int(pts[:, 0].min()) - 5)
    y0 = max(0, int(pts[:, 1].min()) - 5)
    x1 = int(pts[:, 0].max()) + 6
    y1 = int(pts[:, 1].max()) + 6
    h, w = y1 - y0, x1 - x0
    mask = np.zeros((h, w), dtype=np.uint8)
    local_pts = (pts - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(mask, [local_pts], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    high = (dist >= dist.max() * 0.45).astype(np.uint8)
    n_labels, _ = cv2.connectedComponents(high)
    return (n_labels - 1) >= 2


def split_granular_1to2(points, force=False):
    """将一个颗粒拆分为两个，并验证拆分合理性。force=True 时跳过双峰检查（用于长细比极高的 blob）"""
    if not is_split_granular(points):
        return np.array([]), np.array([]), []
    # 仅对极凸形状（sr > 0.90）做双峰检查：防止单颗细长豆被平切假颈误拆
    # sr ≤ 0.90 说明轮廓已有明显凹陷（腰部），直接信任颈部拆分
    if not force and _shape_ratio(points) > 0.90 and not _has_two_peaks(points):
        return np.array([]), np.array([]), []
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    Dmat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(Dmat, 1)
    idx_diff = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n))
    D2 = np.where(idx_diff > n//2, n - idx_diff, idx_diff)
    D2[D2 == 0] = 1
    conv_ratio = Dmat / D2
    mask = (idx_diff < 3) | (idx_diff > n - 3)
    conv_ratio[mask] = np.inf
    min_idx = np.unravel_index(np.argmin(conv_ratio), conv_ratio.shape)
    split_idx = np.sort(min_idx)
    part1, part2 = split_granular(points, split_idx)
    min_val = [Dmat[split_idx[0], split_idx[1]], D2[split_idx[0], split_idx[1]]]

    # 拆分后验证1：若任意一半形状系数 < 0.65，是把单颗细长豆误切成两半，放弃拆分
    # 单豆被横切后各半形状系数约 0.4-0.6；真实粘连豆拆后各自约 0.80-0.95
    # 例外：面积 > 30000 的部分是多豆团，形状不规则属正常，不做此验证
    if part1.shape[0] >= 3 and part2.shape[0] >= 3:
        a1_check = cal_area(part1) < 30000 and _shape_ratio(part1) < 0.65
        a2_check = cal_area(part2) < 30000 and _shape_ratio(part2) < 0.65
        if a1_check or a2_check:
            return np.array([]), np.array([]), []

    # 拆分后验证2：若任意一半面积过小，说明是把一颗豆切成了两小半，放弃拆分
    if cal_area(part1) < 7000 or cal_area(part2) < 7000:
        return np.array([]), np.array([]), []

    return part1, part2, min_val


def _split_by_distance_peaks(contour, min_area=7000, min_sr=0.55):
    """
    适用于近凸形多豆聚簇（sr>0.85）：
    通过距离变换找到各豆的中心区域，将轮廓点分配给最近的中心，
    在分配切换处切割轮廓，得到各子轮廓。
    直接在原轮廓点上分割，不产生奇形轮廓。
    """
    pts = contour.reshape(-1, 2).astype(float)
    n = len(pts)
    if n < 20:
        return []

    pad = 5
    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = int(pts[:, 0].max()) + pad + 1
    y1 = int(pts[:, 1].max()) + pad + 1
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return []

    # 填充轮廓为掩膜
    mask = np.zeros((h, w), dtype=np.uint8)
    local_pts = (pts - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(mask, [local_pts], -1, 255, -1)

    # 距离变换：高值区域 = 豆的核心
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() < 1:
        return []

    # 取距离变换 > 40% 最大值的区域，再做连通域标记
    high = (dist > dist.max() * 0.40).astype(np.uint8)
    n_labels, label_map = cv2.connectedComponents(high)

    if n_labels - 1 < 2:  # 只有1个连通域，无法拆分
        return []

    # 计算每个标签区域的质心（豆子中心，本地坐标系）
    centers = []
    for lbl in range(1, n_labels):
        ys, xs = np.where(label_map == lbl)
        if len(xs) > 0:
            centers.append(np.array([xs.mean(), ys.mean()], dtype=float))

    if len(centers) < 2:
        return []

    centers = np.array(centers)  # (n_peaks, 2) 本地坐标

    # 将轮廓点（本地坐标）分配给距离最近的中心
    pts_local = pts - np.array([x0, y0])
    diff = pts_local[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_to_centers = np.sqrt((diff ** 2).sum(axis=2))  # (n, n_peaks)
    assignments = np.argmin(dists_to_centers, axis=1)    # (n,)

    # 找分配切换点（轮廓点从一个中心跳到另一个中心的位置）
    boundaries = [i for i in range(n)
                  if assignments[i] != assignments[(i + 1) % n]]

    if len(boundaries) < 2:
        return []

    # 提取各子轮廓弧段（边界点 → 下一个边界点之间的轮廓点）
    pieces = []
    nb = len(boundaries)
    for i in range(nb):
        a = boundaries[i]
        b = boundaries[(i + 1) % nb]
        if b > a:
            piece = pts[a:b + 1]
        else:
            piece = np.vstack([pts[a:], pts[:b + 1]])
        if len(piece) < 3:
            continue
        if cal_area(piece) >= min_area and _shape_ratio(piece) >= min_sr:
            pieces.append(piece)

    return pieces if len(pieces) >= 2 else []


def _neck_split_forced(contour, min_sr=0.55, min_area=7000):
    """
    无条件拆分（参考 GranularRecon.split_granular_1to2）：
    找轮廓上 conv_ratio = Dmat/D2 最小的点对作为切割处。
    不做任何前置条件检查，但验证结果：
      - 两半面积均 ≥ min_area
      - 两半 shape_ratio 均 ≥ min_sr（保证是类豆形状，不产生奇形轮廓）
    返回 (part1, part2) 或 ([], []) 表示失败。
    """
    pts = contour.reshape(-1, 2)
    n = pts.shape[0]
    if n < 10:
        return np.array([]), np.array([])

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    Dmat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(Dmat, 1)
    idx_diff = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n))
    D2 = np.where(idx_diff > n // 2, n - idx_diff, idx_diff)
    D2[D2 == 0] = 1
    conv_ratio = Dmat / D2
    # 要求两切点之间的弧长距离 ≥ n/6：
    # 真正的两点跨越相当一段轮廓（每颗豆约占轮廓 1/3），
    # 而轮廓上的微小凹陷两点弧长极短，不是真正的最小处。
    min_arc = max(3, n // 6)
    mask = (idx_diff < min_arc) | (idx_diff > n - min_arc)
    conv_ratio[mask] = np.inf

    split_idx = np.sort(np.unravel_index(np.argmin(conv_ratio), conv_ratio.shape))
    part1, part2 = split_granular(pts, split_idx)

    if part1.shape[0] < 3 or part2.shape[0] < 3:
        return np.array([]), np.array([])
    if cal_area(part1) < min_area or cal_area(part2) < min_area:
        return np.array([]), np.array([])
    if _shape_ratio(part1) < min_sr or _shape_ratio(part2) < min_sr:
        return np.array([]), np.array([])
    return part1, part2


def _force_split_mesh31(contour, max_neck_px, min_sr=0.40, min_area=7000, top_k=30):
    """
    mesh=31 专用切割（用户建议）：
    连通区域（颈部）的最小像素宽度 = 弧长距离 ≥ n//20 的轮廓点对中，
    欧氏距离最小值。若该最小宽度 < max_neck_px，则在最窄处切割。

    与 _neck_split_forced 的区别：
      1. 按欧氏距离排序（直接表征颈宽），而非 conv_ratio
      2. 弧长下限放宽到 n//20（支持 C 形聚簇中弧长较短的颈部）
      3. min_sr 降低到 0.40（切割后一半可能仍是多豆聚簇，允许继续迭代）
      4. 尝试 top_k 个候选（应对全局最窄处验证失败的情形）
    """
    pts = contour.reshape(-1, 2)
    n = pts.shape[0]
    if n < 10:
        return np.array([]), np.array([])

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    Dmat = np.sqrt(np.sum(diff ** 2, axis=2))
    idx_diff = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n))

    # 弧长下限 n//20：过滤紧邻点（面积太小的微小切口会被后续验证自然淘汰）
    min_arc = max(5, n // 20)
    valid = (idx_diff >= min_arc) & (idx_diff <= n - min_arc)
    Dmat_v = np.where(valid, Dmat, np.inf)
    # 只保留上三角，消除对称重复候选
    Dmat_v[np.tril_indices(n)] = np.inf

    flat_sorted = np.argsort(Dmat_v.ravel())
    count = 0
    for flat_idx in flat_sorted:
        if count >= top_k:
            break
        d = Dmat_v.ravel()[flat_idx]
        if d == np.inf or d >= max_neck_px:
            break  # 已超出阈值，后续只会更宽
        i, j = int(np.unravel_index(flat_idx, Dmat_v.shape)[0]), \
                int(np.unravel_index(flat_idx, Dmat_v.shape)[1])
        part1, part2 = split_granular(pts, (i, j))
        if part1.shape[0] < 3 or part2.shape[0] < 3:
            count += 1
            continue
        if cal_area(part1) < min_area or cal_area(part2) < min_area:
            count += 1
            continue
        if _shape_ratio(part1) < min_sr or _shape_ratio(part2) < min_sr:
            count += 1
            continue
        return part1, part2
        count += 1
    return np.array([]), np.array([])


def _erode_and_reconstruct(contour, img_processed, min_sr=0.50, min_area=7000, median_area=None):
    """
    腐蚀重建法：腐蚀深色像素掩膜直到分裂，然后按腐蚀核心到原始暗像素的 Voronoi 分区
    重建各豆掩膜（每个暗像素只归属最近的腐蚀核心，无重叠），再用 findContours 重新找各豆轮廓。
    分界线为两豆腐蚀核心之间的 Voronoi 垂直平分线，即两豆真正的接触切线/分界。
    """
    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 20:
        return []

    pad = 10
    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = min(img_processed.shape[1], int(pts[:, 0].max()) + pad + 1)
    y1 = min(img_processed.shape[0], int(pts[:, 1].max()) + pad + 1)
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return []

    # 构建深色像素掩膜
    cmask = np.zeros((h, w), dtype=np.uint8)
    lp = (pts - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(cmask, [lp], -1, 255, -1)
    crop = img_processed[y0:y1, x0:x1]
    dark_mask = np.zeros((h, w), dtype=np.uint8)
    dark_mask[(crop < COLOR_THRESHOLD) & (cmask > 0)] = 255

    if int(np.sum(dark_mask > 0)) < 1000:
        return []

    # 逐步腐蚀直到深色区域分裂为 2+ 个连通域
    split_radius = None
    split_label_map = None
    split_valid = None
    # 最大腐蚀半径：固定上限与 est_radius 的 90% 取较大值（支持大型近圆聚簇）
    _dark_px_pre = int(np.sum(dark_mask > 0))
    _est_r_pre = np.sqrt(_dark_px_pre / np.pi) if _dark_px_pre > 0 else 50
    _max_erode = max(100, int(_est_r_pre * 0.9))
    for radius in range(2, _max_erode + 1, 2):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        eroded = cv2.erode(dark_mask, kernel)
        if int(np.sum(eroded > 0)) == 0:
            break
        n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(eroded)
        valid = [l for l in range(1, n_labels) if stats[l, cv2.CC_STAT_AREA] >= 300]
        if len(valid) >= 2:
            split_radius = radius
            split_label_map = label_map
            split_valid = valid
            break

    if split_radius is None:
        return []

    # 腐蚀半径过大 → 切到了豆自身，不是真正的颈部，放弃拆分
    # 例外：面积 > 1.5× 中位值时必然是多豆聚簇，允许更深腐蚀
    dark_px_count = int(np.sum(dark_mask > 0))
    est_radius = np.sqrt(dark_px_count / np.pi)
    is_clearly_oversized = median_area and cal_area(pts) > median_area * 1.5
    if not is_clearly_oversized and split_radius > est_radius * 0.4:
        return []

    # 原始 blob 面积 < 1.3× 中位值 → 极可能是单颗异形豆（sr<0.60 触发），放弃拆分
    if median_area and cal_area(pts) < median_area * 1.3:
        return []

    # 小块判定阈值：提高到 0.65× 中位值，防止把单颗豆切成两半后各自通过门槛
    big_thresh = max(min_area, median_area * 0.65) if median_area else min_area

    # 第一步：收集每个腐蚀分量的质心（局部坐标），作为 Voronoi 分配的种子点
    eroded_centroids = []
    for lbl in split_valid:
        ys_e, xs_e = np.where(split_label_map == lbl)
        if len(xs_e) == 0:
            continue
        eroded_centroids.append(np.array([xs_e.mean(), ys_e.mean()], dtype=float))

    if len(eroded_centroids) < 2:
        return []

    # 第二步：确定分界线，将所有暗像素分配给两个分量
    #   两分量时：颈部两尖端点连线（垂直于两豆连线方向的极端点）
    #   三分量及以上：Voronoi（按最近腐蚀质心）
    ys_all, xs_all = np.where(dark_mask > 0)
    if len(xs_all) == 0:
        return []

    centroids_arr = np.array(eroded_centroids, dtype=float)  # (K, 2)
    pixels_xy = np.stack([xs_all, ys_all], axis=1).astype(float)  # (N, 2)

    use_tip_split = False
    if len(eroded_centroids) == 2:
        ca, cb = centroids_arr[0], centroids_arr[1]
        d_ab = cb - ca
        d_len = np.linalg.norm(d_ab)
        if d_len > 1e-6:
            d_norm = d_ab / d_len
            perp = np.array([-d_norm[1], d_norm[0]])  # 垂直于两豆连线

            # 找颈部重叠区域（两膨胀分量的交集）
            dilate_k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * split_radius + 1, 2 * split_radius + 1)
            )
            s0 = ((split_label_map == split_valid[0]).astype(np.uint8)) * 255
            s1 = ((split_label_map == split_valid[1]).astype(np.uint8)) * 255
            dil0 = cv2.bitwise_and(cv2.dilate(s0, dilate_k), dark_mask)
            dil1 = cv2.bitwise_and(cv2.dilate(s1, dilate_k), dark_mask)
            neck = cv2.bitwise_and(dil0, dil1)
            n_ys, n_xs = np.where(neck > 0)

            if len(n_xs) >= 2:
                neck_pts = np.stack([n_xs, n_ys], axis=1).astype(float)
                # 投影到垂直方向，取两个极端点（贴背景的"尖端"）
                proj = neck_pts @ perp
                tip1 = neck_pts[np.argmax(proj)]
                tip2 = neck_pts[np.argmin(proj)]
                line_dir = tip2 - tip1
                if np.linalg.norm(line_dir) > 0:
                    # 叉积：(tip2-tip1) × (pixel-tip1)，正负决定哪侧
                    diff = pixels_xy - tip1
                    cross = line_dir[0] * diff[:, 1] - line_dir[1] * diff[:, 0]
                    # 确认腐蚀质心 0 在哪侧
                    diff_c0 = ca - tip1
                    cross_c0 = line_dir[0] * diff_c0[1] - line_dir[1] * diff_c0[0]
                    final_masks = [np.zeros((h, w), dtype=np.uint8),
                                   np.zeros((h, w), dtype=np.uint8)]
                    sel0 = cross >= 0 if cross_c0 >= 0 else cross < 0
                    sel1 = ~sel0
                    if sel0.any(): final_masks[0][ys_all[sel0], xs_all[sel0]] = 255
                    if sel1.any(): final_masks[1][ys_all[sel1], xs_all[sel1]] = 255
                    use_tip_split = True

    # 第三步：三分量及以上，或尖端分割退化时，用 Voronoi 兜底
    if not use_tip_split:
        chunk = 20000
        assignments = np.empty(len(pixels_xy), dtype=np.int32)
        for start in range(0, len(pixels_xy), chunk):
            end = min(start + chunk, len(pixels_xy))
            d2 = ((pixels_xy[start:end, np.newaxis, :] - centroids_arr[np.newaxis, :, :]) ** 2).sum(axis=2)
            assignments[start:end] = np.argmin(d2, axis=1)
        final_masks = [np.zeros((h, w), dtype=np.uint8) for _ in eroded_centroids]
        for comp_idx in range(len(eroded_centroids)):
            sel = assignments == comp_idx
            if sel.any():
                final_masks[comp_idx][ys_all[sel], xs_all[sel]] = 255

    # 第四步：区分"有效大块"和"小块"
    big_masks, big_centroids = [], []
    small_masks, small_centroids = [], []
    for mask in final_masks:
        sub_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not sub_contours:
            continue
        sub_c = max(sub_contours, key=cv2.contourArea)
        sub_pts = sub_c.reshape(-1, 2).astype(float) + np.array([x0, y0])
        area = cal_area(sub_pts)
        cx, cy = sub_pts[:, 0].mean(), sub_pts[:, 1].mean()
        if area >= big_thresh and _shape_ratio(sub_pts) >= min_sr:
            big_masks.append(mask)
            big_centroids.append(np.array([cx, cy]))
        else:
            small_masks.append(mask)
            small_centroids.append(np.array([cx, cy]))

    # 第五步：把小块的暗像素沿"两颗豆质心连线的垂直平分线"分给两侧的豆
    if small_masks and big_masks:
        big_centroids_arr = np.array(big_centroids, dtype=float)
        for sm, sc in zip(small_masks, small_centroids):
            ys, xs = np.where(sm > 0)
            if len(xs) == 0:
                continue
            dists_to_big = np.sqrt(((big_centroids_arr - sc) ** 2).sum(axis=1))
            sorted_idx = np.argsort(dists_to_big)
            if len(big_masks) >= 2:
                ca = big_centroids_arr[sorted_idx[0]]
                cb = big_centroids_arr[sorted_idx[1]]
                d = cb - ca
                mid = (ca + cb) / 2.0
                pixels = np.stack([xs, ys], axis=1).astype(float)
                dot = ((pixels - mid) * d).sum(axis=1)
                mask_a = np.zeros_like(sm)
                mask_b = np.zeros_like(sm)
                mask_a[ys[dot <= 0], xs[dot <= 0]] = 255
                mask_b[ys[dot >  0], xs[dot >  0]] = 255
                big_masks[sorted_idx[0]] = cv2.bitwise_or(big_masks[sorted_idx[0]], mask_a)
                big_masks[sorted_idx[1]] = cv2.bitwise_or(big_masks[sorted_idx[1]], mask_b)
            else:
                big_masks[0] = cv2.bitwise_or(big_masks[0], sm)

    # 第六步：对合并后的大块重新找轮廓
    pieces = []
    for merged in big_masks:
        sub_contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not sub_contours:
            continue
        sub_c = max(sub_contours, key=cv2.contourArea)
        sub_pts = sub_c.reshape(-1, 2).astype(float) + np.array([x0, y0])
        area = cal_area(sub_pts)
        if area >= min_area and _shape_ratio(sub_pts) >= min_sr:
            pieces.append(sub_pts)

    return pieces if len(pieces) >= 2 else []


#%% 后处理
def postprocess_contours(contours, circle=None, img_processed=None):
    """后处理轮廓：粘连拆分"""
    # 预估单颗豆中位面积：只用 is_split_granular=False 的轮廓（形态正常的孤立豆），
    # 排除碎片（<10000）和明显多豆聚簇，保证 pre_median 准确反映单颗豆大小
    _single_areas = []
    for _c in contours:
        if _c.shape[0] < 3:
            continue
        _pts = _c.reshape(-1, 2).astype(float)
        _a = cal_area(_pts)
        if _a < 10000:
            continue
        if not is_split_granular(_pts):
            _single_areas.append(_a)
    pre_median = float(np.median(_single_areas)) if len(_single_areas) >= 5 else None

    # 第一遍：用形状/颈部条件拆分明显粘连
    # 颈部拆分失败时，用连接桥腐蚀法兜底（适用于圆形花瓣形/侧面相切）
    results = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        smoothed = smooth_granular(contour)
        area_s = cal_area(smoothed)
        # 面积 < 1.3× 预估中位值：极可能是单颗异形豆，跳过第一遍拆分
        if is_split_granular(smoothed) and (pre_median is None or area_s >= pre_median * 1.3):
            stack = [smoothed]
            while stack:
                current = stack.pop(0)
                part1, part2, _ = split_granular_1to2(current)
                if part1.size == 0:
                    p1, p2 = _neck_split_forced(current)
                    if p1.size > 0:
                        for part in [p1, p2]:
                            # 切割后子块面积不足 0.6× 中位值 → 撤销，保留原块
                            if pre_median and cal_area(part) < pre_median * 0.6:
                                results.append(current)
                                break
                            elif is_split_granular(part):
                                stack.append(part)
                            else:
                                results.append(part)
                    else:
                        results.append(current)
                else:
                    # 切割后任一子块面积 < 0.6× 中位值 → 撤销，保留原块
                    a1, a2 = cal_area(part1), cal_area(part2)
                    if pre_median and (a1 < pre_median * 0.6 or a2 < pre_median * 0.6):
                        results.append(current)
                    else:
                        if not is_split_granular(part1):
                            results.append(part1)
                        else:
                            stack.append(part1)
                        if not is_split_granular(part2):
                            results.append(part2)
                        else:
                            stack.append(part2)
        else:
            results.append(smoothed)

    # 第二遍：两豆侧面相切/花瓣形/桥接形，用分水岭或连接桥腐蚀拆分
    # 触发条件：(1) 面积 > 1.5x 中位值（明显过大）
    #           (2) 面积 ≥ 7000 且 shape_ratio < 0.60（明显凹陷，可能是残片或多豆桥接）
    valid_areas = [cal_area(c) for c in results if cal_area(c) >= 7000]
    if len(valid_areas) >= 5:
        median_area = float(np.median(valid_areas))
        area_threshold = median_area * 1.5
        for _ in range(6):  # 最多迭代6次，直到收敛
            next_pass = []
            split_any = False
            for c in results:
                pts = c.reshape(-1, 2) if c.ndim != 2 else c
                area = cal_area(pts)
                sr = _shape_ratio(pts)
                # 面积过大：要求 sr < 0.90（近圆形单豆不拆）
                # 形状系数过低：面积 ≥ 7000 且明显凹陷
                should_try = (area > area_threshold and sr < 0.95) or (area >= 7000 and sr < 0.60)
                if should_try:
                    # 第一优先：腐蚀重建（小块自动合并到邻近大块，不产生圆圈伪影）
                    if img_processed is not None:
                        pieces = _erode_and_reconstruct(c, img_processed, median_area=median_area)
                        if pieces:
                            next_pass.extend(pieces)
                            split_any = True
                            continue
                    # 腐蚀重建失败：颈部切割
                    p1, p2 = _neck_split_forced(c)
                    if p1.size > 0:
                        next_pass.extend([p1, p2])
                        split_any = True
                        continue
                    # 近圆形多豆聚簇：距离峰值切割
                    if sr > 0.85:
                        pieces = _split_by_distance_peaks(c)
                        if pieces:
                            next_pass.extend(pieces)
                            split_any = True
                            continue
                    # 全部失败：形状系数 < 0.50 → 丢弃
                    if sr < 0.50:
                        split_any = True
                        continue
                next_pass.append(c)
            results = next_pass
            if not split_any:
                break

    # 最终检查：长细比 >= 2.5 的 blob 可能是两颗端对端粘连的豆，强制颈部拆分
    final_results = []
    for c in results:
        a = cal_area(c)
        if a >= 7000:
            try:
                ell = cv2.fitEllipse(c.reshape(-1, 1, 2).astype(np.float32))
                aspect = max(ell[1]) / min(ell[1]) if min(ell[1]) > 0 else 1.0
            except Exception:
                aspect = 1.0
            if aspect >= 2.5:
                p1, p2, _ = split_granular_1to2(c, force=True)
                if p1.size > 0:
                    final_results.extend([p1, p2])
                    continue
        final_results.append(c)
    results = final_results

    # 最终检查：目数=31（短轴>11.91mm）的 blob 必然粘连，强制拆分
    # 循环直到没有新的拆分发生（处理拆分后子块仍然≥31的情况）
    if circle is not None and circle[2] > 0:
        pixel_to_mm = 175.0 / (circle[2] * 2)
        mesh31_px = 11.91 / pixel_to_mm  # 超过此像素短轴 = 目数31

        for _ in range(6):  # 最多迭代6次，直到收敛
            # 每轮重新计算正常豆子中位面积（排除当前仍≥31的异常blob）
            normal_areas = [cal_area(c.reshape(-1, 2) if c.ndim != 2 else c)
                            for c in results
                            if (cal_area(c.reshape(-1, 2) if c.ndim != 2 else c) >= 7000
                                and calculate_short_axis_robust(
                                    c.reshape(-1, 2) if c.ndim != 2 else c
                                ) <= mesh31_px)]
            median_area = float(np.median(normal_areas)) if normal_areas else 15000.0

            mesh31_results = []
            any_split = False
            for c in results:
                pts = c.reshape(-1, 2) if c.ndim != 2 else c
                sa = calculate_short_axis_robust(pts)
                if sa > mesh31_px:
                    # 第一优先：conv_ratio 颈部切割
                    p1, p2 = _neck_split_forced(c)
                    if p1.size == 0:
                        # 第二优先（用户建议）：颈宽阈值切割
                        # 阈值 = 4mm，适合粘连咖啡豆颈部宽度
                        max_neck_px = 4.0 / pixel_to_mm
                        p1, p2 = _force_split_mesh31(c, max_neck_px)
                    if p1.size == 0 and img_processed is not None:
                        # 第三优先：腐蚀分割法（逐步腐蚀深色像素区域直到分裂）
                        pieces = _erode_and_reconstruct(c, img_processed, median_area=median_area)
                        if pieces:
                            mesh31_results.extend(pieces)
                            any_split = True
                            continue
                    if p1.size == 0:
                        # 第四优先：距离变换峰值切割（近凸形聚簇）
                        if _shape_ratio(pts) > 0.85:
                            pieces = _split_by_distance_peaks(c)
                            if pieces:
                                mesh31_results.extend(pieces)
                                any_split = True
                                continue
                    if p1.size > 0:
                        mesh31_results.extend([p1, p2])
                        any_split = True
                        continue
                mesh31_results.append(c)
            results = mesh31_results
            if not any_split:
                break

    return results

#%% 主函数
def granular_recon(img_cv):
    """
    主函数：处理图像并返回颗粒特征

    参数:
        img_cv: OpenCV读取的BGR图像

    返回:
        circle: ROI圆参数 (x, y, radius)
        Ginfos_copy_sort: 按面积排序的特征矩阵
        short_axes_pixel: 短轴直径列表（像素单位）
    """
    # 1. 图像预处理
    img_processed, circle = preprocess_img(img_cv)

    # 2. 二值化和轮廓检测
    _, binary = cv2.threshold(img_processed, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = 255 - binary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 3. 后处理和特征提取
    valid_contours = postprocess_contours(contours, circle=circle, img_processed=img_processed)

    # 3b. 过滤含大量空白或多颗豆的轮廓（切割残留的三角形/奇形 piece）：
    #     fill_ratio < 0.55 且内部有 2+ 个暗像素连通域 → 奇形切割残片，丢弃
    filtered_contours = []
    pad3 = 3
    for c in valid_contours:
        pts3 = c.reshape(-1, 2) if c.ndim != 2 else c
        area3 = cal_area(pts3)
        if area3 < 7000:
            filtered_contours.append(c)
            continue
        x0 = max(0, int(pts3[:, 0].min()) - pad3)
        y0 = max(0, int(pts3[:, 1].min()) - pad3)
        x1 = min(img_processed.shape[1], int(pts3[:, 0].max()) + pad3 + 1)
        y1 = min(img_processed.shape[0], int(pts3[:, 1].max()) + pad3 + 1)
        h3, w3 = y1 - y0, x1 - x0
        if h3 <= 0 or w3 <= 0:
            filtered_contours.append(c)
            continue
        cmask3 = np.zeros((h3, w3), dtype=np.uint8)
        lp3 = (pts3 - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(cmask3, [lp3], -1, 255, -1)
        crop3 = img_processed[y0:y1, x0:x1]
        dark3 = np.zeros((h3, w3), dtype=np.uint8)
        dark3[(crop3 < COLOR_THRESHOLD) & (cmask3 > 0)] = 255
        dark_px3 = int(np.sum(dark3 > 0))
        fill3 = dark_px3 / area3 if area3 > 0 else 1.0
        # fill < 0.55：轮廓内超过 45% 是空白/背景，判定为奇形切割残片，丢弃
        if fill3 < 0.55:
            continue
        filtered_contours.append(c)
    valid_contours = filtered_contours

    # 3c. 颈部小圆直接保留，原样显示（不做合并/拆分）

    # 4. 计算特征
    features = []
    short_axes_pixel = []
    for contour in valid_contours:
        if contour.shape[0] > 2:
            feature_vector = get_granular_info(contour)
            features.append(feature_vector)
            # 用健壮版短轴（二次验证防弧形轮廓虚高）
            pts = contour.reshape(-1, 2) if contour.ndim != 2 else contour
            short_axes_pixel.append(calculate_short_axis_robust(pts))

    if not features:
        return circle, np.array([]), []

    # 5. 转换为numpy数组并计算比表面积
    features_np = np.array(features)
    specific_surface = features_np[:, 4] / features_np[:, 1]
    features_np = np.column_stack([features_np, specific_surface])

    # 6. 按面积排序（从大到小）
    sort_idx = np.argsort(features_np[:, 1])[::-1]
    Ginfos_copy_sort = features_np[sort_idx]

    # 7. 按相同顺序排序短轴直径和轮廓
    short_axes_pixel_sorted = [short_axes_pixel[i] for i in sort_idx]
    valid_contours_sorted = [valid_contours[i] for i in sort_idx]

    # 8. 修正含空白区域的轮廓短轴：
    #    RETR_EXTERNAL 轮廓会将 C 形聚簇中间的空白也围入其中，
    #    导致 shoelace 面积/fitEllipse 短轴虚高，目数误判为 31。
    #    对于填充率 < 0.88 的 blob（轮廓内含大量空白），
    #    改用深色像素的内切圆直径代替 fitEllipse 短轴。
    if circle[2] > 0:
        pixel_to_mm_local = 175.0 / (circle[2] * 2)
        mesh31_px_local = 11.91 / pixel_to_mm_local
        pad = 3
        for k in range(len(valid_contours_sorted)):
            if short_axes_pixel_sorted[k] <= mesh31_px_local:
                continue  # 短轴已在正常范围，无需修正
            pts = valid_contours_sorted[k]
            pts = pts.reshape(-1, 2) if pts.ndim != 2 else pts
            filled_area = cal_area(pts)
            if filled_area < 5000:
                continue
            x0 = max(0, int(pts[:, 0].min()) - pad)
            y0 = max(0, int(pts[:, 1].min()) - pad)
            x1 = min(img_processed.shape[1], int(pts[:, 0].max()) + pad + 1)
            y1 = min(img_processed.shape[0], int(pts[:, 1].max()) + pad + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            h, w = y1 - y0, x1 - x0
            cmask = np.zeros((h, w), dtype=np.uint8)
            lp = (pts - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(cmask, [lp], -1, 255, -1)
            crop = img_processed[y0:y1, x0:x1]
            dark_mask = np.zeros((h, w), dtype=np.uint8)
            dark_mask[(crop < COLOR_THRESHOLD) & (cmask > 0)] = 255
            dark_area = int(np.sum(dark_mask > 0))
            fill_ratio = dark_area / filled_area if filled_area > 0 else 1.0
            if fill_ratio < 0.88:
                dist_dark = cv2.distanceTransform(dark_mask, cv2.DIST_L2, 5)
                dark_inscribed_diam = float(dist_dark.max() * 2)
                if dark_inscribed_diam > 0:
                    short_axes_pixel_sorted[k] = min(short_axes_pixel_sorted[k], dark_inscribed_diam)

    return circle, Ginfos_copy_sort, short_axes_pixel_sorted, valid_contours_sorted, img_processed


#%% 检测结果可视化
def draw_detection_result(img_processed, circle, valid_contours, mesh_numbers, save_path=None):
    """
    在预处理图（3500×3500）上绘制检测结果并保存。
    轮廓坐标与 img_processed 同属 3500px 坐标系，无需缩放。

    参数:
        img_processed: granular_recon 返回的预处理灰度图（3500×3500）
        circle: ROI 圆参数 (x, y, radius)，3500px 坐标系
        valid_contours: 有效轮廓列表，3500px 坐标系
        mesh_numbers: 每个轮廓对应的目数列表
        save_path: 保存路径，None 则不保存

    返回:
        result_img: 绘制结果的 BGR 图像
    """
    # -- 目数→颜色映射（绿→黄→红，对应小目→大目）--
    def mesh_to_color(mesh):
        ratio = max(0.0, min(1.0, (mesh - 12) / (31 - 12)))
        if ratio < 0.5:
            r, g, b = int(255 * ratio * 2), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - ratio) * 2), 0
        return (b, g, r)  # OpenCV BGR

    # 灰度图转 BGR 作为绘制底图
    result_img = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)

    # 绘制 ROI 圆
    cv2.circle(result_img,
               (int(circle[0]), int(circle[1])), int(circle[2]),
               (200, 200, 200), 4)

    # 绘制每个豆子的轮廓和目数标签（坐标已在 3500px 空间，直接使用）
    # mesh_numbers 与 valid_contours 一一对齐，None 表示该轮廓无法分配目数
    font = cv2.FONT_HERSHEY_SIMPLEX
    for contour, mesh in zip(valid_contours, mesh_numbers):
        pts = contour.astype(np.int32)
        color = mesh_to_color(mesh) if mesh is not None else (255, 255, 255)

        # 轮廓线
        cv2.polylines(result_img, [pts.reshape(-1, 1, 2)], True, color, 3)

        if mesh is None:
            continue  # 无目数轮廓只画边框，不贴标签（低于12目下限）

        # 目数标签置于质心
        cx_b = int(np.mean(pts[:, 0]))
        cy_b = int(np.mean(pts[:, 1]))
        label = str(mesh)
        (tw, th), _ = cv2.getTextSize(label, font, 1.2, 2)
        cv2.rectangle(result_img,
                      (cx_b - tw // 2 - 4, cy_b - th - 4),
                      (cx_b + tw // 2 + 4, cy_b + 4),
                      color, -1)
        cv2.putText(result_img, label,
                    (cx_b - tw // 2, cy_b),
                    font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # 缩小到 1200px 宽便于查看（原图 3500px 太大）
    out_w = 1200
    out_h = int(result_img.shape[0] * out_w / result_img.shape[1])
    result_img = cv2.resize(result_img, (out_w, out_h))

    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, result_img)

    return result_img


def draw_mesh_distribution(x_data, y_data, y_accumulate, save_path=None, title="咖啡豆目数分布"):
    """
    绘制目数分布图（柱状图 + 累计折线），双 Y 轴帕累托风格。

    参数
    ----
    x_data       : list[float]  各目数
    y_data       : list[float]  各目数占比 %
    y_accumulate : list[float]  累计占比 %
    save_path    : str | None   保存路径（None 则只返回不保存）
    title        : str          图表标题

    返回
    ----
    fig : matplotlib Figure 对象
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import os

    if not x_data:
        return None

    x = list(x_data)
    y = list(y_data)
    y_acc = list(y_accumulate)
    x_labels = [str(int(v)) for v in x]

    fig, ax1 = plt.subplots(figsize=(max(8, len(x) * 0.7), 5))

    # 柱状图（左轴：占比 %）
    bar_color = "#4C8CBF"
    bars = ax1.bar(x_labels, y, color=bar_color, alpha=0.85, zorder=2, label="占比 %")
    ax1.set_xlabel("目数", fontsize=12)
    ax1.set_ylabel("占比 (%)", fontsize=12, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.set_ylim(0, max(y) * 1.3 if y else 10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # 在柱子顶部标数值
    for bar, val in zip(bars, y):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=bar_color)

    # 折线图（右轴：累计占比 %）
    ax2 = ax1.twinx()
    line_color = "#E05C2A"
    ax2.plot(x_labels, y_acc, color=line_color, marker="o", linewidth=2,
             markersize=5, zorder=3, label="累计 %")
    ax2.set_ylabel("累计占比 (%)", fontsize=12, color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # 在折线点旁标数值
    for xi, ya in zip(x_labels, y_acc):
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

    return fig
