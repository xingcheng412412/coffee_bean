import logging
import time
import httpx
import io
from PIL import Image
import hashlib


def resize_image(image, short_side_length=0, long_side_length=0):
    """
    对图片进行resize，保持纵横比，可以选择以短边或长边为基准进行缩放。
    
    参数:
    image: PIL Image 对象。
    short_side_length: 目标短边长度，默认为0（不按短边缩放）。
    long_side_length: 目标长边长度，默认为0（不按长边缩放）。
    
    返回:
    resize 后的 PIL Image 对象。如果不需要resize，则返回原始 Image 对象。
    
    注意:
    - 如果同时指定了short_side_length和long_side_length，将优先使用short_side_length。
    - 如果两个参数都为0，则不进行缩放，直接返回原图。
    """
    width, height = image.size
    
    # 如果两个参数都为0，不进行缩放
    if short_side_length == 0 and long_side_length == 0:
        return image
    
    # 优先按短边缩放
    if short_side_length > 0:
        short_side = min(width, height)
        # 如果短边已经小于等于目标长度，无需resize
        if short_side <= short_side_length:
            return image
        # 计算缩放比例
        scale_factor = short_side_length / short_side
    # 按长边缩放
    elif long_side_length > 0:
        long_side = max(width, height)
        # 如果长边已经小于等于目标长度，无需resize
        if long_side <= long_side_length:
            return image
        # 计算缩放比例
        scale_factor = long_side_length / long_side
    
    # 计算新的尺寸
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # 进行resize
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # 使用 LANCZOS 算法获得较好的resize质量
    logging.info(f"图片已调整大小至: {new_width}x{new_height}")
    return resized_image

def image_preprocess(file_obj, filename, short_side_length=0, long_side_length=0):
    logging.info(f"开始使用解析图片: {filename}..")
    try:
        original_bytes = None
        if isinstance(file_obj, str) and (file_obj.startswith("http://") or file_obj.startswith("https://")):
            logging.info(f"开始下载图片文件: {file_obj}..")
            download_start_time = time.time()
            with httpx.Client() as client:
                response = client.get(file_obj)
                response.raise_for_status()  # 确保请求成功
                original_bytes = response.content
                file_obj = io.BytesIO(original_bytes)
            download_time = time.time() - download_start_time
            logging.info(f"图片文件下载完成: {filename}.. 下载时间: {download_time:.2f}秒")
        else:
            # file_obj为文件对象或BytesIO
            if hasattr(file_obj, 'read'):
                # 读取全部内容并重置指针
                original_bytes = file_obj.read()
                file_obj.seek(0)
            else:
                # 直接是bytes
                original_bytes = file_obj
                file_obj = io.BytesIO(original_bytes)
        # 计算md5
        md5 = hashlib.md5(original_bytes).hexdigest()
        image = Image.open(file_obj)
        image = resize_image(image, short_side_length, long_side_length)
        return image, md5
    except Exception as e:
        logging.error(f"解析图片文件失败: {e}")
        raise
