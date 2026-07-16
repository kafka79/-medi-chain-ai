import sys
import os
import asyncio
import logging
import threading
import signal

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variable so the imported API module doesn't start internal loops in lifespan
os.environ["RUN_BACKGROUND_WORKERS_IN_API"] = "false"

# Perform import after environment variables are set
from deployment.api.main import cleanup_old_temp_files, reconcile_dlq_task, logger

# Keep track of loops and tasks for graceful shutdown
running_tasks = []
running_loops = []

def _shutdown_loop_gracefully(loop, task):
    try:
        loop.call_soon_threadsafe(task.cancel)
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")

def run_cleanup():
    logger.info("[Worker Thread] Starting temp file cleanup worker...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    running_loops.append(loop)
    task = loop.create_task(cleanup_old_temp_files())
    running_tasks.append((loop, task))
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        logger.info("[Worker Thread] Cleanup task cancelled gracefully.")
    except Exception as e:
        logger.error(f"[Worker Thread] Cleanup task encountered error: {e}")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        if hasattr(loop, 'shutdown_default_executor'):
            loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

def run_dlq_reconciliation():
    logger.info("[Worker Thread] Starting DLQ reconciliation worker...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    running_loops.append(loop)
    task = loop.create_task(reconcile_dlq_task())
    running_tasks.append((loop, task))
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        logger.info("[Worker Thread] DLQ task cancelled gracefully.")
    except Exception as e:
        logger.error(f"[Worker Thread] DLQ task encountered error: {e}")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        if hasattr(loop, 'shutdown_default_executor'):
            loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

import argparse

def main():
    parser = argparse.ArgumentParser(description="MEdi Chain background worker process.")
    parser.add_argument(
        "--task",
        choices=["cleanup", "dlq", "all"],
        default="all",
        help="Specify the background task to execute. Run 'all' (default) to start both tasks in thread-isolated loops."
    )
    args = parser.parse_args()

    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()
        for loop, task in running_tasks:
            _shutdown_loop_gracefully(loop, task)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.task == "cleanup":
        logger.info("Starting standalone temp file cleanup worker (process-isolated)...")
        t1 = threading.Thread(target=run_cleanup, name="cleanup-worker", daemon=False)
        t1.start()
        t1.join()
    elif args.task == "dlq":
        logger.info("Starting standalone DLQ reconciliation worker (process-isolated)...")
        t2 = threading.Thread(target=run_dlq_reconciliation, name="dlq-worker", daemon=False)
        t2.start()
        t2.join()
    else:
        logger.warning(
            "WARNING: Running both background tasks in a single process. "
            "For production deployments, split them into process-isolated services using '--task cleanup' and '--task dlq'."
        )
        logger.info("Starting standalone background worker process with thread-isolated event loops...")
        t1 = threading.Thread(target=run_cleanup, name="cleanup-worker", daemon=False)
        t2 = threading.Thread(target=run_dlq_reconciliation, name="dlq-worker", daemon=False)
        
        t1.start()
        t2.start()
        
        try:
            while not shutdown_event.is_set():
                t1.join(timeout=1.0)
                t2.join(timeout=1.0)
                if not t1.is_alive() and not t2.is_alive():
                    break
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt in main thread.")
            signal_handler(signal.SIGINT, None)

        t1.join()
        t2.join()

if __name__ == "__main__":
    main()
