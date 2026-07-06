import sys
import os
import asyncio
import logging
import threading

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variable so the imported API module doesn't start internal loops in lifespan
os.environ["RUN_BACKGROUND_WORKERS_IN_API"] = "false"

# Perform import after environment variables are set
from deployment.api.main import cleanup_old_temp_files, reconcile_dlq_task, logger

def run_cleanup():
    logger.info("[Worker Thread] Starting temp file cleanup worker...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(cleanup_old_temp_files())
    except Exception as e:
        logger.error(f"[Worker Thread] Cleanup task encountered error: {e}")
    finally:
        loop.close()

def run_dlq_reconciliation():
    logger.info("[Worker Thread] Starting DLQ reconciliation worker...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(reconcile_dlq_task())
    except Exception as e:
        logger.error(f"[Worker Thread] DLQ task encountered error: {e}")
    finally:
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
    
    if args.task == "cleanup":
        logger.info("Starting standalone temp file cleanup worker (process-isolated)...")
        run_cleanup()
    elif args.task == "dlq":
        logger.info("Starting standalone DLQ reconciliation worker (process-isolated)...")
        run_dlq_reconciliation()
    else:
        logger.warning(
            "WARNING: Running both background tasks in a single process. "
            "For production deployments, split them into process-isolated services using '--task cleanup' and '--task dlq'."
        )
        logger.info("Starting standalone background worker process with thread-isolated event loops...")
        t1 = threading.Thread(target=run_cleanup, name="cleanup-worker", daemon=True)
        t2 = threading.Thread(target=run_dlq_reconciliation, name="dlq-worker", daemon=True)
        
        t1.start()
        t2.start()
        
        try:
            while True:
                t1.join(timeout=1.0)
                t2.join(timeout=1.0)
                if not t1.is_alive() and not t2.is_alive():
                    break
        except KeyboardInterrupt:
            logger.info("Received SIGINT, exiting worker process.")

if __name__ == "__main__":
    main()
