#!/usr/bin/env python
import sys
sys.path.append("..") # 添加项目根目录, 可以找到common目录下的公共代码.
import time
import logging
import asyncio
import threading
import math
import traceback
from datetime import datetime, timedelta
import httpx
import hashlib
import json

from celery import Celery
from settings import CELERY_BROKER_URL, CELERY_WORKER_CONCURRENCY

log_format = "%(asctime)s | %(process)d | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

logging.warning("celery.particle_callback connect to : %s" % (CELERY_BROKER_URL))
celery_app = Celery('particle_callback', broker=CELERY_BROKER_URL)

celery_app.conf.update({
    'worker_hijack_root_logger': False,
    'worker_redirect_stdouts': False,
    'worker_redirect_stdouts_level': 'INFO',
    'worker_log_format': log_format,
    'worker_task_log_format': log_format,
    'accept_content': ['json'],
    'task_serializer': 'json',
    'result_serializer': 'json',
    'broker_login_method': 'PLAIN',
    'worker_prefetch_multiplier': 1,
    'task_protocol': 1,
})
logging.warning("celery.particle_callback connect to : %s finished." %( CELERY_BROKER_URL ))

# --- Callback constants and functions ---
CALLBACK_RETRY_COUNT = 5
STATUS_FINISHED = 2
STATUS_ERROR = 3

def eta_second(delay_second):
    return datetime.utcnow() + timedelta(seconds=delay_second)

def httpx_post(url, body, headers, timeout, ignoreError=False):
    headers_str = " ".join([f"-H '{k}: {v}'" for k, v in headers.items()])
    body_str = json.dumps(body, ensure_ascii=False)
    req_debug = f"curl -X POST {headers_str} -d '{body_str}' '{url}'"
    try:
        with httpx.Client() as client:
            resp = client.post(url, data=body_str, headers=headers, timeout=timeout)
        
        if resp.status_code == 200:
            logging.info(f"callback [{req_debug}] success! status: {resp.status_code}")
            return True, resp.status_code, resp.text
        else:
            logging.error(f"callback [{req_debug}] failed! status: {resp.status_code}, reason: {resp.reason_phrase}")
            return False, resp.status_code, "callback failed! " + str(resp.reason_phrase)
    except Exception as e:
        logging.error(f"callback [{req_debug}] exception: {e}")
        return False, 500, f"callback exception: {e}"


def particle_callback_sign(body):
    task_id = body['task_id']
    magic = "Ai%ZNkwXyOXu2zY$"
    sign_str = magic + str(task_id)
    signature = hashlib.sha1(sign_str.encode('utf-8')).hexdigest()

    suffix_magic = "i6t,IJYuHA"
    suffix_str = suffix_magic + signature
    suffix = hashlib.sha1(suffix_str.encode('utf-8')).hexdigest()[0:3]
    signature = signature + "." + suffix
    return signature

def sync_particle_callback(task):
    try:
        callback_url = task['callback_url']
        body = task['body']
        task_id = body['task_id']
        body['sign'] = particle_callback_sign(body)
        ignoreError = task.get("ignoreError", False)
        timeout = task.get("timeout", 5)
        headers = {"Content-Type": "application/json"}
        callback_count = task.get("callback_count", 1)

        ok, status_code, resp = httpx_post(callback_url, body, headers, timeout, ignoreError=ignoreError)
        if ok:
            logging.info(f"Task {task_id} callback success. resp: {resp}")
        else:
            if int(status_code / 100) == 5:
                callback_finished = callback_count > CALLBACK_RETRY_COUNT
            else:
                callback_finished = True # 其它状态码, 不再重试
            errmsg = str(resp)

            logging.error(f"Task {task_id} callback failed. errmsg={errmsg}, callback_count={callback_count}")
            if not callback_finished:
                # 重新发送任务.
                delay_second = int(math.pow(2, callback_count))
                callback_count += 1
                task["callback_count"] = callback_count
                particle_callback_task.apply_async(args=[task], eta=eta_second(delay_second))
                logging.warning(f"Rescheduling callback for task {task_id} in {delay_second} seconds.")
    except Exception as err:
        logging.error("sync_particle_callback(%s) catch exception: [%s]" % (task, traceback.format_exc()))

@celery_app.task(name='particle_callback_task')
def particle_callback_task(task):
    sync_particle_callback(task)

# --- End of callback functions ---

class ParticleCallbackWorkerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(log_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        concurrency = CELERY_WORKER_CONCURRENCY
        worker = celery_app.Worker(
            concurrency=concurrency,
            loglevel='INFO',
            queues=['particle_callback'] # Consume from a specific queue
        )
        worker.start()


def start_particle_callback_worker(join=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logging.info("Preparing to start particle callback celery worker thread...")
    celeryWorkerThread = ParticleCallbackWorkerThread()
    celeryWorkerThread.start()
    logging.warning("start particle callback celeryWorkerThread ...")
    if join:
        celeryWorkerThread.join()

if __name__ == "__main__":
    start_particle_callback_worker(join=True)