import os
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("storage-provider")

class StorageProvider(ABC):
    @abstractmethod
    def save(self, data, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def delete(self, path: str):
        pass

    @abstractmethod
    def cleanup(self, max_age_seconds: int):
        pass

class LocalStorageProvider(StorageProvider):
    """Addresses the 'Stateful Temp Storage' flaw with a cleaner abstraction."""
    def __init__(self, root_dir: str = "temp/storage"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, file_obj, relative_path: str):
        dest = self.root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file_obj, f)
        return str(dest)

    def load(self, relative_path: str):
        return str(self.root / relative_path)

    def delete(self, relative_path: str):
        path = self.root / relative_path
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()

    def cleanup(self, max_age_seconds: int):
        try:
            import time
            now = time.time()
            if self.root.exists():
                for item in self.root.iterdir():
                    is_uuid_dir = len(item.name) == 32 and all(c in "0123456789abcdefABCDEF" for c in item.name)
                    if item.is_dir() and is_uuid_dir and (now - item.stat().st_mtime > max_age_seconds):
                        self.delete(item.name)
                        logger.info(f"Cleaned up old local storage directory: {item.name}")
        except Exception as e:
            logger.error(f"Failed to cleanup local storage: {e}")

class S3StorageProvider(StorageProvider):
    """MinIO/S3 implementation to resolve the 'Stateful Temp Storage' flaw for K8s."""
    def __init__(self, endpoint: str = None, access_key: str = None, secret_key: str = None, bucket: str = "medi-chain-bucket"):
        from minio import Minio
        # Default fallback to standard docker-compose environment variables or local minio
        self.endpoint = endpoint or os.getenv("S3_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("S3_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("S3_SECRET_KEY", "minioadmin")
        self.bucket = bucket
        
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=False
            )
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except Exception as e:
            logger.error(f"S3StorageProvider initialization error: {e}")
            self.client = None

    def save(self, file_obj, relative_path: str):
        if not self.client:
            logger.error("S3 client not initialized. Cannot save.")
            raise RuntimeError("S3 client not initialized")
            
        try:
            import os
            from tempfile import NamedTemporaryFile
            
            # Copy stream to a temp file to determine file size for MinIO put_object
            with NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(file_obj, tmp)
                tmp_path = tmp.name
                
            file_size = os.path.getsize(tmp_path)
            with open(tmp_path, "rb") as f:
                self.client.put_object(
                    self.bucket,
                    relative_path,
                    f,
                    length=file_size
                )
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
                
            logger.info(f"Successfully uploaded to s3://{self.bucket}/{relative_path}")
            return relative_path
        except Exception as e:
            logger.error(f"Failed to save file to S3: {e}")
            raise

    def load(self, relative_path: str):
        if not self.client:
            logger.error("S3 client not initialized. Cannot load.")
            raise RuntimeError("S3 client not initialized")
            
        try:
            from tempfile import NamedTemporaryFile
            suffix = Path(relative_path).suffix
            # Download file locally for models to read
            tmp = NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_close_name = tmp.name
            tmp.close()
            
            self.client.fget_object(self.bucket, relative_path, tmp_close_name)
            logger.info(f"Downloaded s3://{self.bucket}/{relative_path} to {tmp_close_name}")
            return tmp_close_name
        except Exception as e:
            logger.error(f"Failed to load file from S3: {e}")
            raise

    def delete(self, relative_path: str):
        if not self.client:
            logger.error("S3 client not initialized. Cannot delete.")
            return
            
        try:
            # If relative_path is a directory prefix (e.g. UUID request ID), list and delete recursively
            objects = self.client.list_objects(self.bucket, prefix=relative_path, recursive=True)
            for obj in objects:
                self.client.remove_object(self.bucket, obj.object_name)
            self.client.remove_object(self.bucket, relative_path)
            logger.info(f"Deleted prefix s3://{self.bucket}/{relative_path}")
        except Exception as e:
            logger.error(f"Failed to delete file from S3: {e}")

    def cleanup(self, max_age_seconds: int):
        if not self.client:
            return
            
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            objects = self.client.list_objects(self.bucket, recursive=True)
            for obj in objects:
                age = (now - obj.last_modified).total_seconds()
                if age > max_age_seconds:
                    self.client.remove_object(self.bucket, obj.object_name)
                    logger.info(f"Cleaned up old object from S3: {obj.object_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup S3 bucket: {e}")
