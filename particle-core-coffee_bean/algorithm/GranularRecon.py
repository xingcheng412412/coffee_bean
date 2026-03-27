
#%% 本程序用于通过图像计算颗粒粒径分布

#%% import 

import numpy as np
import matplotlib.pyplot as plt
import cv2

#%% 配置常量
# 输出/裁切后图像目标尺寸（最终标准化为正方形）
IMAGE_SIZE = 1750 * 2
# 二值化阈值（用于将灰度图转换为前景/背景）。可根据图像调整或改为自适应。
COLOR_THRESHOLD = 230

#%% 预处理、ROI 相关工具函数
def get_circle(center, R):
    """返回圆周上的点（用于可视化）。

    参数:
        center: (x,y) 圆心
        R: 半径
    返回:
        (x, y) 两个数组表示圆周坐标
    """
    angs = np.linspace(0, 2*np.pi, 1000)
    x = R * np.cos(angs) + center[0]
    y = R * np.sin(angs) + center[1]
    return  (x, y)

#%% 载入并预处理图像，检测圆形 ROI 并标准化裁切
def preprocess_img(img_cv):
    """对输入的彩色图像进行灰度化、缩放、圆形 ROI 检测和背景去除。

    输入:
        img_cv: OpenCV 读入的图像（通常为 BGR,但函数使用 COLOR_RGB2GRAY,注意读取一致性）
    输出:
        img_cat_: 裁切并标准化到 IMAGE_SIZE 的灰度图（用于后续分割）
        circle: 圆心与半径（已放缩到 IMAGE_SIZE 的坐标系）
    步骤简介:
        1. 转灰度并按最短边等比例放大到 IMAGE_SIZE
        2. 缩小(10 倍)后用 HoughCircles 寻找圆盘（定位样品区域）
        3. 构建圆形掩膜，将圆外像素替换为圆内低位灰度的 5 百分位值，减小背景影响
        4. 用高斯滤波构造局部背景 Gimg,并用 img_/Gimg 做局部归一化
        5. 裁切出圆的内接矩形并将边缘外 20 px 置为白，最后 resize 到 IMAGE_SIZE
    """
    # 转为灰度
    img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 按图像较短边放缩到目标尺寸比例
    size_ = min([img.shape[0], img.shape[1]])
    ratio_ = IMAGE_SIZE / size_
    shape_ = [int(img.shape[1]*ratio_), int(img.shape[0]*ratio_)]
    img = cv2.resize(img, shape_)

    # 缩小图像用于圆盘检测
    tmp = cv2.resize(img, (int(img.shape[1]/10), int(img.shape[0]/10)))
    _, binary = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找最大白色连通区域，拟合最小外接圆
    contours_roi, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_roi:
        raise ValueError("未检测到圆形 ROI，请检查图像或调整参数")
    largest = max(contours_roi, key=cv2.contourArea)
    (cx_small, cy_small), r_small = cv2.minEnclosingCircle(largest)
    circle = np.array([cx_small * 10, cy_small * 10, r_small * 10])

    n_ = img.shape
    rect_ = np.array([[circle[0]-circle[2], circle[0]+circle[2]], [circle[1]-circle[2], circle[1]+circle[2]]], dtype=np.int64)

    # 构建圆形掩膜：保留圆内像素；圆外用圆内像素的 5% 分位替代（去背景）
    mx, my = np.meshgrid(np.arange(n_[1]), np.arange(n_[0]))
    mask = np.sqrt((np.square((mx - circle[0])) + np.square((my - circle[1])))) < (circle[2])
    vtmp1 = img[ mask]
    img_ = img.copy()
    img_[~mask] = np.percentile(vtmp1, 5)
    
    # 大尺度高斯平滑用于局部亮度归一化（缩4x计算后放大，速度提升约50x）
    _scale = 4
    _small = cv2.resize(img_, (img_.shape[1]//_scale, img_.shape[0]//_scale))
    _Gsmall = cv2.GaussianBlur(_small.astype(np.float32) / 255.0, (0, 0), 30 // _scale)
    Gimg = cv2.resize(_Gsmall, (img_.shape[1], img_.shape[0]))
    

    # 局部归一化并裁切出圆内矩形区域（clamp 边界防止越界）
    img_new = img_.astype(np.float64) / 255.0 / Gimg * 1
    img_new[img_new > 1] = 1.0
    img_new[img_new < 0] = 0.0
    img_new = np.array(img_new * 255, dtype=np.uint8)
    x0 = int(max(0, rect_[0, 0])); x1 = int(min(img.shape[1], rect_[0, 1]))
    y0 = int(max(0, rect_[1, 0])); y1 = int(min(img.shape[0], rect_[1, 1]))
    img_cat = img_new[y0:y1, x0:x1]

    # 将圆边缘外略微扩展的区域置白，防止裁切后带入外环噪声
    crop_h, crop_w = img_cat.shape[:2]
    cx_crop = int(circle[0]) - x0
    cy_crop = int(circle[1]) - y0
    r_crop  = int(circle[2])
    mx, my = np.meshgrid(np.arange(crop_w), np.arange(crop_h))
    mask = np.sqrt((mx - cx_crop)**2 + (my - cy_crop)**2) > (r_crop - 20)
    img_cat_ = img_cat.copy()
    img_cat_[mask] = 255

    # 最后将裁切区域 resize 到固定 IMAGE_SIZE 大小并返回
    circle = (circle * IMAGE_SIZE / img_cat_.shape[0]).astype(np.int32) 
    img_cat_ = cv2.resize(img_cat_, (IMAGE_SIZE ,IMAGE_SIZE))

    return img_cat_, circle

#%% 在图像上填充轮廓（用于可视化或标记）
def contours_draw_img(contours,p,v): 
    """在图像 p 上填充轮廓（用于可视化或标记）"""
    cv2.drawContours(p, [contours], -1, v, thickness=-1)
    return p

#%% 廓形滤波与去锯齿：对轮廓点做一维平滑
def mean_3(vec, N):
    N2 = int(N/2)
    vecc = np.zeros_like(vec)
    len = vec.shape[0]
    for i in range(N2+1):
        vecc[i] = np.mean(vec[:i+N2+1])
        vecc[-i-1] = np.mean(vec[-i-N2-1:])
    for i in range(N2, len-N2-1):
        vecc[i+1] = vecc[i] + (vec[i+N2+1]-vec[i-N2])/(N+1)
    return vecc 

def get_mean(vec, N, NUM):
    """重复应用 mean_3 实现更高阶的平滑"""
    N = 2 * int(N/2)
    if N == 0 or NUM == 0:
        return vec 
    for _ in range(NUM):
        vec = mean_3(vec, N)
    return vec  

def get_mean_mat(mat,N,NUM):
    for i in range(mat.shape[1]):
        mat[:, i] = get_mean(mat[:, i], N, NUM)
    return mat

def smooth_granular(gra):
    """对单个轮廓坐标做平滑处理，返回平滑后的坐标数组"""
    gra = np.squeeze(np.double(gra))
    if len(gra) > 2:
        n = gra.shape[0]; n2 = int(n/2)
        gra = np.vstack((gra, gra[1:, :])) 
        gra = get_mean_mat(gra, 2, 1)
        gra = gra[n2+1:n2+n, :]
    return gra

#%% 面积/周长 等几何计算
def cross(v1, v2):
    return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

def cal_area(GRAi):
    """基于三角剖分计算多边形面积（以第一个点为基准）"""
    ptr_xys = GRAi
    ptr_ori = GRAi[0] 
    v1s = ptr_xys[0:-2, :] - ptr_ori
    v2s = ptr_xys[1:-1, :] - ptr_ori 
    val = cross(v1s, v2s)
    return np.abs(np.sum(val) / 2)

# 优化 cal_perimeter 函数
def cal_perimeter(GRAi):
    """使用向量化计算周长"""
    if GRAi.shape[0] < 2:
        return 0
    diffs = np.diff(GRAi, axis=0, prepend=GRAi[-1:])
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

def get_granular_info(GRAi): 
    """计算单个颗粒的几何特征并返回字典:
    center_pix, 像素周长, 实际面积, 凸包面积, 形状系数, 周长, 凸包周长, 中心坐标, 凸包点集
    """
    """计算单个颗粒的几何特征 - 使用OpenCV凸包算法"""
    center_pix = np.mean(GRAi, axis=0)
    
    # 去除重复点（可选，但通常OpenCV convexHull可以处理）
    if len(GRAi.shape) == 2 and GRAi.shape[1] == 2:
        # 转换为 (n, 1, 2) 格式
        points = GRAi.reshape(-1, 1, 2).astype(np.float32)
    else:
        # 如果已经是正确格式，直接使用
        points = GRAi.astype(np.float32)

    # 使用OpenCV的凸包算法
    try:
        hull = cv2.convexHull(points, returnPoints=True)
        # 转换为 (m, 2) 格式
        IJs_edge_cov = hull.reshape(-1, 2)
    except Exception as e:
        print(f"凸包计算失败，使用原始点: {e}")
        IJs_edge_cov = np.unique(GRAi, axis=0)  # 回退方案

    # 周长与面积计算
    perimeter_conv = cal_perimeter(IJs_edge_cov)
    perimeter = cal_perimeter(GRAi) 
    convex_area = cal_area(IJs_edge_cov) 
    gra_area = cal_area(GRAi)

    pix_perimeter = GRAi.shape[0]
    if (gra_area < 1) or (convex_area < 1):
        # 若面积异常小，则用像素数作为回退值
        gra_area = pix_perimeter
        convex_area = pix_perimeter
        perimeter = pix_perimeter
        perimeter_conv = pix_perimeter

    info = {}
    info['center_pix'] = center_pix 
    info['pix_perimeter'] = pix_perimeter
    info['gra_area'] = gra_area
    info['convex_area'] = convex_area
    info['shape_ratio'] = gra_area / convex_area
    info['perimeter'] = perimeter
    info['perimeter_conv'] = perimeter_conv
    info['vector'] = [pix_perimeter, gra_area, convex_area, info['shape_ratio'], perimeter, perimeter_conv, center_pix[0], center_pix[1]]
    info['IJs_edge_cov'] = IJs_edge_cov 

    return info

#%% 颗粒拆分相关函数（用于处理粘连颗粒）
def is_split_granular(GRAi):
    """判断是否需要拆分：依据面积与形状系数阈值"""
    info = get_granular_info(GRAi)
    if (info['gra_area'] > 4) and (info['shape_ratio'] < 0.88):
        return True
    else:
        return False 

def smooth_granular_face(Gra1, ind):
    """在分割边界处做局部平滑，减小锯齿"""
    Gra1_ = Gra1 
    ind3 = np.array([-1, 0, 1], dtype=np.int32)
    iters = ind + np.array([-1, 0, 1, 2], dtype=np.int32)
    for i in iters:
        Gra1_[i, 0] = np.mean(Gra1[i+ind3, 0])
        Gra1_[i, 1] = np.mean(Gra1[i+ind3, 1])
    return Gra1_

def split_granular(GRAi_, ix):
    """按索引区间 ix 将轮廓环拆成两段，并做边界平滑"""
    ind1 = np.hstack([np.arange(0, ix[0]+1), np.arange(ix[1], GRAi_.shape[0]), 0])
    ind2 = np.arange(ix[0], ix[1]+1)
    Gra1 = GRAi_[ind1, :]
    Gra2 = GRAi_[ind2, :]
    try:
        if Gra1.shape[0] > 4:
            Gra1_ = np.vstack([Gra1[-3:, :], Gra1, Gra1]) 
            Gra1_ = smooth_granular_face(Gra1_, ix[0]+3)
            Gra1 = Gra1_[4:Gra1.shape[0]+4, :]
        if Gra2.shape[0] > 4:
            Gra2_ = np.vstack([Gra2[-6:, :], Gra2]); 
            Gra2_ = smooth_granular_face(Gra2_, 4)
            Gra2 = Gra2_[0:Gra2.shape[0]+1, :]
    except:
        print('split failed!')
    return Gra1, Gra2

def split_granular_1to2(GRAi_):
    """尝试将一个可能粘连的轮廓拆分为两个：
    方法：计算所有点对的欧氏距离 Dmat 与沿轮廓索引距离 D2,
    以 conv_ratio = Dmat / D2 的最小值对应点对作为切割处（启发式）。
    返回两个子轮廓和最小距离指标。
    """
    if not is_split_granular(GRAi_):
        return np.array([]), np.array([]), []

    Dmat = np.sqrt(np.square(GRAi_[:, 0] - GRAi_[:, 0][:, np.newaxis]) + \
        np.square(GRAi_[:, 1] - GRAi_[:, 1][:, np.newaxis]))
    Dmat[Dmat == 0] = 1
    n = GRAi_.shape[0]
    n2 = int(n/2)
    vecs = np.arange(n)
    D0 = vecs[:, np.newaxis] - vecs
    D1 = np.abs(D0)

    bo = D1 > n2 
    D2 = D1
    D2[bo] = n - D1[bo]
    D2[D2 == 0] = 1

    mask  = np.bitwise_or(D1 < 3, D1 > n-3)
    conv_ratio = Dmat / D2
    conv_ratio[mask] = np.nan 
    min_conv = np.nanmin(conv_ratio)

    ixy = np.argwhere(conv_ratio == min_conv) 
    ix = [ixy[0][0], ixy[0][1]]
    ix = np.sort(ix)

    min_dist = Dmat[ix[0], ix[1]]
    min_ldis = D2[ix[0], ix[1]]

    Gra1, Gra2 = split_granular(GRAi_, ix)   
    min_val = [min_dist, min_ldis]

    return Gra1, Gra2, min_val 

#%% 银皮（亮薄膜）判定——用于剔除噪声性薄膜区域
def is_grind_skin(feature):
    # feature: [像素数, 均值, std, pct15, pct50, pct85]
    b1 = np.bitwise_and((feature[0] > 200), (feature[1] > 0.48))
    b2 = np.bitwise_and((feature[0] > 30) , (feature[1] > 0.62))
    return np.bitwise_or(b1, b2)

#%% 后处理：对所有初始颗粒进行剔除银皮与粘连拆分的递归处理
def postprocess_GRAs(GRAs): 
    GRAs_copy = []
    TmpGRAs = []
    counter_nonconv = 0
    counter_skin = 0

    for i in range(len(GRAs)):
        # 先剔除被判为银皮的轮廓
        if is_grind_skin(GRAs[i][1]):
            counter_skin += 1
            continue

        GRAi = GRAs[i][0]
        Gra1, Gra2, min_val = split_granular_1to2(GRAi)

        if len(min_val) == 0:
            # 无需拆分，直接加入结果
            GRAs_copy.append(GRAi) 
        else:
            # 拆分成功，分别检查子片是否仍需继续拆分（递归队列）
            counter_nonconv += 1
            if not is_split_granular(Gra1):
                GRAs_copy.append(Gra1)
            else:
                TmpGRAs.append(Gra1)
            if not is_split_granular(Gra2):
                GRAs_copy.append(Gra2)
            else:
                TmpGRAs.append(Gra2)

            while len(TmpGRAs) > 0:
                GRAi = TmpGRAs.pop(0)
                Gra1, Gra2, min_val = split_granular_1to2(GRAi)
                if Gra1.size == 0:
                    GRAs_copy.append(GRAi) 
                else:
                    counter_nonconv += 1
                    if not is_split_granular(Gra1):
                        GRAs_copy.append(Gra1)
                    else:
                        TmpGRAs.append(Gra1)
                    if not is_split_granular(Gra2):
                        GRAs_copy.append(Gra2)
                    else:
                        TmpGRAs.append(Gra2)
    return GRAs_copy, counter_nonconv, counter_skin

# %% 主流程：颗粒重构与特征提取
def granular_recon(img_cv):
    """入口函数:输入彩色图像,返回颗粒轮廓、裁切图、ROI、特征表等

    返回:
        GRAs_copy: 拆分与过滤后的轮廓列表
        img_cat_: 预处理后用于分割的图像
        circle: ROI 圆参数 (x,y,r)
        contours: 原始 findContours 返回的轮廓
        Ginfos_copy_sort: 按实际面积排序的特征矩阵
        feature_title: 特征列名
        counter_nonconv: 非凸拆分计数
        counter_skin: 被判为银皮并剔除的计数
    """
    img_cat_, circle = preprocess_img(img_cv)

    # 固定阈值二值化（前景为颗粒），然后反转使前景为白色
    _, data = cv2.threshold(img_cat_, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY)
    data = 255-data

    # 识别外轮廓（每个外轮廓代表一个颗粒或颗粒簇）
    contours, hierarchy = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 统计每个轮廓内部像素的亮度特征（用 bounding box 小掩码代替全图标号图，速度提升约20x）
    nGRAs = len(contours)
    GRAs_ = []
    for i in range(nGRAs):
        x, y, w, h = cv2.boundingRect(contours[i])
        patch = img_cat_[y:y+h, x:x+w].astype(np.float64) / 255.0
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contours[i] - [x, y]], -1, 1, thickness=-1)
        tmp = patch[mask == 1]
        if tmp.size == 0:
            tmp = np.array([0.0])
        GRAs_.append([
            tmp.size,
            np.mean(tmp),
            np.std(tmp),
            np.percentile(tmp, 15),
            np.percentile(tmp, 50),
            np.percentile(tmp, 85)])

    # 将原始轮廓与统计特征组合，且对轮廓坐标先做平滑
    GRAs = []
    for i in range(nGRAs):
        c = contours[i]
        if c.size > 2:
            GRAs.append([smooth_granular(c), GRAs_[i]])

    # 后处理：剔除银皮并递归拆分粘连颗粒
    GRAs_copy, counter_nonconv, counter_skin = postprocess_GRAs(GRAs)

    print(len(GRAs_copy), len(GRAs)) 

    # 汇总每个最终颗粒的几何特征
    Ginfos_copy = []
    for i in range(len(GRAs_copy)):
        info = get_granular_info(GRAs_copy[i])
        Ginfos_copy.append(info['vector'])

    Ginfos_copy_np = np.array(Ginfos_copy)
    vtmp = Ginfos_copy_np[:, 4] / Ginfos_copy_np[:, 1]
    Ginfos_copy_np = np.hstack([Ginfos_copy_np, vtmp[:, np.newaxis]])

    feature_title = ['像素周长(pix)', '实际面积(pix^2)', '凸包面积(pix^2)', '形状系数', 
        '周长(pix)', '凸包周长(pix)', '几何中心X(pix)', '几何中心Y(pix)', '比表面积(1/pix)']

    feature_title_str = feature_title[0]
    for str_ in feature_title[1:]: 
        feature_title_str += ','
        feature_title_str += str_

    sort_ind = np.argsort(Ginfos_copy_np[:, 1])
    Ginfos_copy_sort = Ginfos_copy_np[sort_ind[::-1], :] 

    return GRAs_copy, img_cat_, circle, contours, Ginfos_copy_sort, feature_title, counter_nonconv, counter_skin