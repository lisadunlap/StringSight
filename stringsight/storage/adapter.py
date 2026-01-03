from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import Any, List
import json
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

def get_storage_adapter() -> StorageAdapter:
    """Factory function to get the configured storage adapter."""
    return LocalFileSystemAdapter()
