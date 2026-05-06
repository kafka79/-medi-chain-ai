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

class S3StorageProvider(StorageProvider):
    """Skeleton for future cloud migration (Fixing 'Stateful Temp Storage' for K8s)."""
    def save(self, data, path: str):
        logger.info(f"[MOCK S3] Uploading to s3://medi-chain-bucket/{path}")
        
    def load(self, path: str):
        return f"s3://medi-chain-bucket/{path}"
        
    def delete(self, path: str):
        logger.info(f"[MOCK S3] Deleting s3://medi-chain-bucket/{path}")
