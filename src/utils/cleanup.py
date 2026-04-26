import os
import time
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_old_sessions(temp_dir="temp/sessions", max_age_seconds=86400):
    """
    Automated TTL Cleanup for session folders.
    Fix for Rohan: Prevents disk bloat in production.
    Default: 24 hours.
    """
    if not os.path.exists(temp_dir):
        return

    now = time.time()
    count = 0
    
    for session_id in os.listdir(temp_dir):
        session_path = os.path.join(temp_dir, session_id)
        if os.path.isdir(session_path):
            # Check the last modification time of the folder
            mtime = os.path.getmtime(session_path)
            if now - mtime > max_age_seconds:
                try:
                    shutil.rmtree(session_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {session_path}: {e}")
                    
    if count > 0:
        logger.info(f"Cleanup complete. Deleted {count} expired sessions.")

if __name__ == "__main__":
    cleanup_old_sessions()
