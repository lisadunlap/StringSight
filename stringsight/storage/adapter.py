import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union
import json
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    class ClientError(Exception):
        pass

from stringsight.config import settings
from stringsight.logging_config import get_logger

logger = get_logger(__name__)

class StorageAdapter(ABC):
    """Abstract base class for storage adapters."""
    
    @abstractmethod
    def ensure_directory(self, path: str) -> None:
        """Ensure a directory exists."""
        pass
    
    @abstractmethod
    def write_text(self, path: str, content: str) -> None:
        """Write text content to a file."""
        pass
    
    @abstractmethod
    def read_text(self, path: str) -> str:
        """Read text content from a file."""
        pass
    
    @abstractmethod
    def write_json(self, path: str, data: Any) -> None:
        """Write JSON data to a file."""
        pass
    
    @abstractmethod
    def read_json(self, path: str) -> Any:
        """Read JSON data from a file."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        pass
    
    @abstractmethod
    def list_files(self, path: str, pattern: str = "*") -> List[str]:
        """List files in a directory matching a pattern."""
        pass

    @abstractmethod
    def write_jsonl(self, path: str, records: List[Any]) -> None:
        """Write a list of records as JSONL (one JSON object per line)."""
        pass

    @abstractmethod
    def read_jsonl(self, path: str) -> List[Any]:
        """Read JSONL file and return list of records."""
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file."""
        pass

    @abstractmethod
    def copy(self, src: str, dst: str) -> None:
        """Copy a file from src to dst."""
        pass

class LocalFileSystemAdapter(StorageAdapter):
    """Adapter for local filesystem storage."""
    
    def ensure_directory(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        
    def write_text(self, path: str, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        
    def read_text(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")
        
    def write_json(self, path: str, data: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    def read_json(self, path: str) -> Any:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
            
    def exists(self, path: str) -> bool:
        return Path(path).exists()
        
    def list_files(self, path: str, pattern: str = "*") -> List[str]:
        p = Path(path)
        if not p.exists():
            return []
        return [str(f) for f in p.glob(pattern)]

    def write_jsonl(self, path: str, records: List[Any]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def read_jsonl(self, path: str) -> List[Any]:
        records = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def delete(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    def copy(self, src: str, dst: str) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

class S3Adapter(StorageAdapter):
    """Adapter for S3-compatible object storage."""
    
    def __init__(self):
        if boto3 is None:
            raise ImportError("boto3 is required for S3Adapter. Please install it with `pip install boto3`.")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        self.bucket = settings.S3_BUCKET
        self._ensure_bucket()
        
    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3.create_bucket(Bucket=self.bucket)
            except Exception as e:
                logger.error(f"Failed to create bucket {self.bucket}: {e}")

    def ensure_directory(self, path: str) -> None:
        # S3 doesn't have directories, but we can create a 0-byte object ending in /
        if not path.endswith("/"):
            path += "/"
        self.s3.put_object(Bucket=self.bucket, Key=path)
        
    def write_text(self, path: str, content: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=content.encode("utf-8"))
        
    def read_text(self, path: str) -> str:
        response = self.s3.get_object(Bucket=self.bucket, Key=path)
        return response["Body"].read().decode("utf-8")
        
    def write_json(self, path: str, data: Any) -> None:
        content = json.dumps(data)
        self.write_text(path, content)
        
    def read_json(self, path: str) -> Any:
        content = self.read_text(path)
        return json.loads(content)
        
    def exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path)
            return True
        except ClientError:
            # Check if it's a "directory"
            if not path.endswith("/"):
                path += "/"
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path)
                return True
            except ClientError:
                return False

    def list_files(self, path: str, pattern: str = "*") -> List[str]:
        # Simple prefix listing, pattern matching is limited
        if not path.endswith("/"):
            path += "/"
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=path)
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]

    def write_jsonl(self, path: str, records: List[Any]) -> None:
        lines = []
        for record in records:
            lines.append(json.dumps(record))
        content = "\n".join(lines) + "\n"
        self.write_text(path, content)

    def read_jsonl(self, path: str) -> List[Any]:
        content = self.read_text(path)
        records = []
        for line in content.split("\n"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def delete(self, path: str) -> None:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=path)
        except ClientError as e:
            logger.warning(f"Failed to delete {path}: {e}")

    def copy(self, src: str, dst: str) -> None:
        copy_source = {"Bucket": self.bucket, "Key": src}
        self.s3.copy_object(CopySource=copy_source, Bucket=self.bucket, Key=dst)

def get_storage_adapter() -> StorageAdapter:
    """Factory function to get the configured storage adapter."""
    if settings.STORAGE_TYPE == "s3":
        return S3Adapter()
    return LocalFileSystemAdapter()
