import logging
import os
import io
import time
import threading
import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache

from utils.image_utils import image_preprocess
from algorithm.particle_go import get_partical
from settings import CACHE_RESULT, MAX_WORKERS
# Configuration from particle-core-service.py

# Global objects for caching and threading
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
detect_cache = LRUCache(maxsize=256) if CACHE_RESULT else None
cache_lock = threading.Lock() if CACHE_RESULT else None

async def detect_particle_from_image(file_obj, filename: str, detect_type: int = 0):
    """
    Detects particles from an image file object.
    This function contains the core logic, independent of web frameworks.
    """
    detect_info = {
        "code": 200,
        "msg": None,
        "data": None,
    }

    try:
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        # Preprocess the image once.
        image, md5 = await loop.run_in_executor(thread_pool, image_preprocess, file_obj, filename)
        image_process_time = round(time.time() - start_time, 2)
        logging.info(f"{filename} 图片预处理: {image_process_time:.2f}秒")

        cache_key = f"{md5}_{detect_type}"

        # Check cache
        if CACHE_RESULT and cache_lock and detect_cache:
            with cache_lock:
                if cache_key in detect_cache:
                    logging.info(f"命中缓存: {cache_key}")
                    detect_info = detect_cache[cache_key]
                    detect_info["stats"] = {
                        "image_process_time": image_process_time,
                        "detection_time": 0,
                    }
                    return detect_info

        # Convert image to file object for get_partical
        processed_file_obj = io.BytesIO()
        image.save(processed_file_obj, format='JPEG')
        processed_file_obj.seek(0)
        
        # Run detection
        start_time = time.time()
        ok, result = get_partical(processed_file_obj, filename=filename, detect_type=detect_type)
        detection_time = round(time.time() - start_time, 2)
        stats = {
            "image_process_time": image_process_time,
            "detection_time": detection_time,
        }
        logging.info(f"{filename} 图片检测: {detection_time:.2f}秒")
        if ok:
            detect_info["code"] = 200
            detect_info["data"] = result.dict()
            detect_info["stats"] = stats
        else:
            detect_info["code"] = 500
            detect_info["msg"] = result
            detect_info["stats"] = stats

        # Write to cache
        if ok and CACHE_RESULT and cache_lock and detect_cache:
            with cache_lock:
                detect_cache[cache_key] = detect_info.copy()

    except Exception as e:
        import traceback
        logging.error(f"Error during particle detection for {filename}: {traceback.format_exc()}")
        detect_info["code"] = 500
        detect_info["msg"] = f"Detection failed: {str(e)}"

    return detect_info

"""
python core/detector.py --file http://s3-test.finishlinetech.cn/mys3/particle/50cecdbb-9fda-421e-8692-1e1bb701ad89/2025/09/17/105433/a5c599a8c8e4ebdeaf8739a49ebb7aaf47636af7.jpg
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Particle detection")
    parser.add_argument("--file", type=str, required=True, help="File to detect")
    parser.add_argument("--type", type=int, default=0, help="Type of detection")
    args = parser.parse_args()
    is_http_url = args.file.startswith("http://") or args.file.startswith("https://")
    file_obj = args.file if is_http_url else open(args.file, "rb")
    filename = args.file
    detect_type = args.type
    resp = asyncio.run(detect_particle_from_image(file_obj, filename, detect_type))
    print(resp)
    