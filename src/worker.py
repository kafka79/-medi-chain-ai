import sys
import os
import asyncio
import logging

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variable so the imported API module doesn't start internal loops in lifespan
os.environ["RUN_BACKGROUND_WORKERS_IN_API"] = "false"

# Perform import after environment variables are set
from deployment.api.main import cleanup_old_temp_files, reconcile_dlq_task, logger

async def main():
    logger.info("Starting standalone MEdi Chain background worker process...")
    try:
        # Run both tasks concurrently
        await asyncio.gather(
            cleanup_old_temp_files(),
            reconcile_dlq_task()
        )
    except asyncio.CancelledError:
        logger.info("Background worker tasks cancelled. Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Unhandled exception in background worker: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received SIGINT, exiting worker process.")
