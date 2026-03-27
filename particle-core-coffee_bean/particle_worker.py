#!/usr/bin/env python
import sys
sys.path.append("..") # 添加项目根目录, 可以找到common目录下的公共代码.
import time
import logging
import asyncio
import threading
import math
import traceback
import signal
from datetime import datetime, timedelta
import httpx
import hashlib
log_format = "%(asctime)s | %(process)d | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
import json
from celery import Celery
from settings import CELERY_BROKER_URL, CELERY_WORKER_CONCURRENCY
from core.detector import detect_particle_from_image


logging.warning("celery.particle connect to : %s" % (CELERY_BROKER_URL))
celery_app = Celery('particle', broker=CELERY_BROKER_URL)

# CELERY_TASK_PROTOCOL 1
# https://docs.celeryproject.org/en/stable/userguide/configuration.html
celery_app.conf.update({
    'worker_hijack_root_logger': False,  # 禁用根日志劫持
    'worker_redirect_stdouts': False,    # 添加这行，防止重定向标准输出
    'worker_redirect_stdouts_level': 'INFO',  # 设置输出级别
    'worker_log_format': log_format,
    'worker_task_log_format': log_format,  # 添加任务日志格式
    'accept_content': ['json'],
    'task_serializer': 'json',
    'result_serializer': 'json',
    'broker_login_method': 'PLAIN',
    'worker_prefetch_multiplier': 1,
    'task_protocol': 1,
    # 优雅退出相关配置
    'worker_disable_rate_limits': True,  # 禁用限流，加快shutdown
    'task_soft_time_limit': 300,  # 软时间限制5分钟
    'task_time_limit': 360,  # 硬时间限制6分钟
    'worker_max_tasks_per_child': 1000,  # 限制子进程任务数，避免内存泄漏
})
logging.warning("celery.particle connect to : %s finished." %( CELERY_BROKER_URL ))

@celery_app.task(name='particle_detect_task')
def particle_detect_task(task):
    """
    Celery task for particle detection.
    """
    task_id = task['task_id']
    url = task['url']
    detect_type = task.get('detect_type',0)
    callback_url = task.get('callback_url')
    call_at = task.get('call_at', int(time.time()))

    logging.info(f"Received particle detection task for task_id: {task_id}, url: '{url}' with detect_type: {detect_type}")
    try:
        # Since detect_particle_from_image is an async function, we need to run it in an event loop.
        result = asyncio.run(detect_particle_from_image(file_obj=url, filename=url, detect_type=detect_type))

        if result and isinstance(result, dict):
            result["task_id"] = task_id
            result["call_at"] = call_at

        logging.info(f"Detection for '{url}' completed. Result: {json.dumps(result, ensure_ascii=False)}")
        body = result
        if callback_url:
            callback_task = {
                "callback_url": callback_url,
                "body": body,
            }
            celery_app.send_task('particle_callback_task', args=[callback_task], queue='particle_callback', eta=eta_second(0.2))
            logging.info(f"Sent callback task for task_id: {task_id} to queue 'particle_callback'")
        else:
            logging.warning(f"No callback url for task_id: {task_id}, url: '{url}' with detect_type: {detect_type}, result: {json.dumps(result, ensure_ascii=False)}")
        return result
    except Exception as e:
        logging.error(f"Error during particle detection for '{url}': {e}", exc_info=True)
        return {"code": 500, "msg": f"An unexpected error occurred in worker: {e}", "data": None}

def eta_second(delay_second):
    return datetime.utcnow() + timedelta(seconds=delay_second)

class ParticleWorkerThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.worker = None
        self.shutdown_event = threading.Event()

    def run(self):
        """
        https://docs.celeryq.dev/en/stable/reference/celery.worker.worker.html
        """
        # 确保日志配置在worker启动前正确设置
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 如果没有handler，添加一个console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(log_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        concurrency = CELERY_WORKER_CONCURRENCY
        self.worker = celery_app.Worker(
            concurrency=concurrency,
            loglevel='INFO',  # 使用字符串形式
            queues=['celery'], # Explicitly consume from the default queue
        )
        # 不使用setup_defaults，让celery使用我们的日志配置
        self.worker.start()

    def stop(self, timeout=30):
        """停止worker，如果超时则强制退出"""
        if self.worker and not self.shutdown_event.is_set():
            logging.warning(f"Gracefully stopping worker with {timeout}s timeout...")
            self.shutdown_event.set()
            # 先尝试优雅退出
            self.worker.stop()
            
    def force_stop(self):
        """强制停止worker"""
        if self.worker:
            logging.warning("Force stopping worker...")
            self.shutdown_event.set()
            self.worker.stop()


def start_particle_worker(join=False):
    # 在启动worker前确保日志配置正确
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 确保有合适的handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logging.info("准备启动particle celery worker线程...")
    celeryWorkerThread = ParticleWorkerThread()
    celeryWorkerThread.start()
    logging.warning("start particle celeryWorkerThread ...")
    if join:
        shutdown_timer = None
        
        def graceful_shutdown(signum, frame):
            nonlocal shutdown_timer
            logging.warning(f"Received signal {signum}, shutting down gracefully...")
            
            # 取消之前的timer（如果存在）
            if shutdown_timer:
                shutdown_timer.cancel()
            
            # 启动优雅退出
            celeryWorkerThread.stop(timeout=15)
            
            # 设置强制退出timer
            def force_shutdown():
                logging.warning("Timeout reached, force stopping worker...")
                celeryWorkerThread.force_stop()
                # 给一点时间让force stop完成，然后退出进程
                def final_exit():
                    logging.warning("Exiting process...")
                    import os
                    os._exit(1)
                
                final_timer = threading.Timer(3.0, final_exit)
                final_timer.start()
            
            shutdown_timer = threading.Timer(15.0, force_shutdown)
            shutdown_timer.start()

        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)

        celeryWorkerThread.join()

if __name__ == "__main__":
    start_particle_worker(join=True)
