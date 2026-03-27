#!/usr/bin/env python
import sys
import time
import logging
import asyncio
import httpx
log_format = "%(asctime)s | %(process)d | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
import json
import uvicorn
from fastapi import FastAPI, Request
from particle_worker import particle_detect_task

# --- FastAPI Server and Task Runner ---

app = FastAPI()

# This will be set to an asyncio.Event in the main execution block.
shutdown_event = None

@app.post("/callback")
async def receive_callback(request: Request):
    """
    Receives the callback, logs it, and triggers server shutdown.
    """
    global shutdown_event
    body = await request.json()
    logging.info(f"Received callback. Body: {json.dumps(body, ensure_ascii=False)}")
    if shutdown_event:
        shutdown_event.set()
    return {"status": "callback received"}

async def run_task_and_server(url: str, detect_type: int = 0):
    """
    Starts a temporary FastAPI server, dispatches a Celery task,
    and waits for a callback or a timeout before shutting down.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=8199, log_level="info")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())

    # The local server is now running, so we can dispatch the task.
    callback_url = "http://127.0.0.1:8199/callback"
    task_id = f"local-task-for-{str(time.time())}"
    task = {"url": url, "detect_type": detect_type, "callback_url": callback_url, "task_id": task_id}
    logging.info(f"Dispatching particle detection task {task}")
    particle_detect_task.delay(task)

    logging.info("Task dispatched. Waiting for callback or timeout (120s)...")
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=120.0)
        logging.info("Callback received. Shutting down server.")
    except asyncio.TimeoutError:
        logging.warning("Timeout of 120s reached without callback. Shutting down server.")
    
    server.should_exit = True
    await server_task
    logging.info("Server shut down.")

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--detect_type', type=int, default=0)
    args = parser.parse_args()

    # Set the global shutdown_event for the callback handler to use.
    global shutdown_event
    shutdown_event = asyncio.Event()
    
    try:
        # The logic from __main__ is now encapsulated in this function,
        # along with the new server functionality.
        await run_task_and_server(args.url, args.detect_type)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
