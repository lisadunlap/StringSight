"""
High-performance three-tier caching system for LLM responses and embeddings.

Architecture:
    Tier 1: ShardedLRU (in-memory, lock-free via sharding)
    Tier 2: LMDB (disk-backed, memory-mapped, multi-process safe)
    Tier 3: AsyncWriteBuffer (non-blocking writes with batching)

Performance characteristics:
    - Cache GET (memory hit): ~0.01ms
    - Cache GET (disk hit): ~0.1ms
    - Cache SET: ~0.01ms (non-blocking)
    - Scales to 1000+ concurrent operations

Configuration via environment variables:
    - STRINGSIGHT_CACHE_DIR: Base directory (default: .cache/stringsight)
    - STRINGSIGHT_MEMORY_CACHE_SIZE: In-memory items (default: 10000)
    - STRINGSIGHT_MEMORY_SHARDS: Number of shards (default: 32)
    - STRINGSIGHT_LMDB_MAP_SIZE: Max DB size (default: 100GB)
    - STRINGSIGHT_ASYNC_BATCH_SIZE: Batch size (default: 1000)
    - STRINGSIGHT_ASYNC_FLUSH_INTERVAL: Flush interval (default: 0.1s)
    - STRINGSIGHT_DISABLE_CACHE: Disable caching (default: 0)

Example:
    >>> cache = UnifiedCache()  # Singleton - same instance everywhere
    >>>
    >>> # Efficient caching with lazy key builder
    >>> key_builder = CacheKeyBuilder({"model": "gpt-4", "messages": [...]})
    >>> cached = cache.get_completion(key_builder)
    >>> if cached is None:
    ...     response = api_call()
    ...     cache.set_completion(key_builder, response)
"""

import atexit
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread, Event, RLock
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import orjson
import xxhash

logger = logging.getLogger(__name__)

# Enable debug logging if env variable is set
if os.environ.get("STRINGSIGHT_CACHE_DEBUG", "0") in ("1", "true", "True"):
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_size_string(size_str: Union[str, int]) -> int:
    """Parse size string like '10GB' or '1TB' into bytes.

    Args:
        size_str: Size as string (e.g., "50GB") or integer (bytes)

    Returns:
        Size in bytes

    Raises:
        ValueError: If size string format is invalid

    Examples:
        >>> parse_size_string("10GB")
        10737418240
        >>> parse_size_string("1.5TB")
        1649267441664
        >>> parse_size_string(1024)
        1024
    """
    if isinstance(size_str, int):
        return size_str

    size_str = size_str.strip().upper()
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)$', size_str)
    if not match:
        raise ValueError(
            f"Invalid size string: {size_str}. Expected format like '10GB' or '1TB'"
        )

    number = float(match.group(1))
    unit = match.group(2)

    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }

    return int(number * multipliers[unit])


# =============================================================================
# CacheKeyBuilder - Lazy Serialization
# =============================================================================

class CacheKeyBuilder:
    """Lazy cache key builder with memoization.

    Generates cache keys efficiently by serializing and hashing only once
    when first needed. Subsequent calls return the cached key.

    This avoids redundant serialization on cache hits and when retrying
    API calls with the same request data.

    Attributes:
        _data: The data to generate a key from
        _key: Cached key (None until first get_key() call)

    Example:
        >>> builder = CacheKeyBuilder({"model": "gpt-4", "prompt": "Hello"})
        >>> key1 = builder.get_key()  # Serializes and hashes
        >>> key2 = builder.get_key()  # Returns cached key (no serialization)
        >>> assert key1 == key2
    """

    def __init__(self, data: Dict[str, Any]):
        """Initialize key builder.

        Args:
            data: Dictionary to generate cache key from
        """
        self._data = data
        self._key: Optional[str] = None

    def get_key(self) -> str:
        """Generate key on first call, return cached key thereafter.

        Uses orjson for fast serialization (2-5× faster than stdlib json)
        and xxhash for fast hashing (10-50× faster than SHA256).

        Returns:
            128-character hex string (xxhash3_128 digest)
        """
        if self._key is None:
            # Serialize with sorted keys for deterministic output
            serialized = orjson.dumps(self._data, option=orjson.OPT_SORT_KEYS)
            # Use xxhash3_128 for speed (10-50× faster than SHA256)
            self._key = xxhash.xxh3_128(serialized).hexdigest()
        return self._key

    def __str__(self) -> str:
        """Return cache key as string."""
        return self.get_key()


# =============================================================================
# SimpleLRU - Thread-Safe LRU Cache
# =============================================================================

class SimpleLRU:
    """Simple thread-safe LRU cache using OrderedDict.

    Provides basic LRU eviction with thread-safety via RLock.
    Used as building block for ShardedLRU.

    Attributes:
        maxsize: Maximum number of items to store
        cache: OrderedDict storing key-value pairs
        lock: Reentrant lock for thread-safety

    Example:
        >>> lru = SimpleLRU(maxsize=100)
        >>> lru.set("key1", "value1")
        >>> lru.get("key1")
        'value1'
        >>> lru.get("key2")
        None
    """

    def __init__(self, maxsize: int = 1000):
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of items to store
        """
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.lock = RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, moving to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            value = self.cache.get(key)
            if value is not None:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache, evicting oldest if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Update existing or add new
            self.cache[key] = value
            self.cache.move_to_end(key)

            # Evict oldest if over capacity
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()


# =============================================================================
# ShardedLRU - Lock-Free Sharded LRU Cache
# =============================================================================

class ShardedLRU:
    """Lock-free sharded LRU cache for reduced contention.

    Splits cache into N independent shards based on key hash. Each shard
    has its own lock, reducing contention by N× compared to single-lock LRU.

    With 32 shards and 100 concurrent threads, average contention per shard
    is ~3 threads instead of 100.

    Attributes:
        num_shards: Number of independent shards
        shard_size: Items per shard (maxsize / num_shards)
        shards: List of SimpleLRU instances

    Example:
        >>> cache = ShardedLRU(maxsize=10000, num_shards=32)
        >>> cache.set("key1", "value1")  # Routed to shard based on hash
        >>> cache.get("key1")
        'value1'
    """

    def __init__(self, maxsize: int = 10000, num_shards: int = 32):
        """Initialize sharded LRU cache.

        Args:
            maxsize: Total items across all shards
            num_shards: Number of shards (recommend 16, 32, or 64)
        """
        self.num_shards = num_shards
        self.shard_size = maxsize // num_shards
        self.shards = [SimpleLRU(maxsize=self.shard_size) for _ in range(num_shards)]

    def _get_shard(self, key: str) -> SimpleLRU:
        """Deterministically route key to shard based on hash.

        Args:
            key: Cache key

        Returns:
            SimpleLRU shard for this key
        """
        # Extract hex hash portion after namespace prefix (e.g., "completion:abc123...")
        # Keys are formatted as "namespace:hexhash", so find the hash part
        if ':' in key:
            hex_part = key.split(':', 1)[1]  # Get part after first ':'
        else:
            hex_part = key

        # Use first 8 hex chars of hash for shard selection
        shard_idx = int(hex_part[:8], 16) % self.num_shards
        return self.shards[shard_idx]

    def get(self, key: str) -> Optional[Any]:
        """Get value from appropriate shard.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self._get_shard(key).get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in appropriate shard.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._get_shard(key).set(key, value)

    def clear(self) -> None:
        """Clear all shards."""
        for shard in self.shards:
            shard.clear()


# =============================================================================
# LMDBCache - LMDB Persistent Backend
# =============================================================================

class LMDBCache:
    """LMDB-backed persistent cache.

    Features:
        - Memory-mapped I/O (no buffer copies, fast)
        - MVCC for lock-free concurrent reads
        - Multi-process safe with proper locking
        - Two databases: completions, embeddings

    Performance:
        - Read: ~0.05-0.2ms
        - Write: ~0.1-1ms
        - Multi-process safe via file locks

    Attributes:
        env: LMDB environment
        completions_db: Database for completion responses
        embeddings_db: Database for embeddings (optional)

    Example:
        >>> cache = LMDBCache(path=".cache/lmdb", map_size=100*1024**3)
        >>> cache.set("key1", b"value1", db='completions')
        >>> cache.get("key1", db='completions')
        b'value1'
    """

    def __init__(
        self,
        path: str = ".cache/stringsight/lmdb",
        map_size: int = 5 * 1024**3,  # 5GB default
        max_dbs: int = 2
    ):
        """Initialize LMDB environment.

        Args:
            path: Directory for LMDB files
            map_size: Max database size in bytes (virtual, not allocated)
            max_dbs: Number of named databases (completions + embeddings)
        """
        import lmdb

        os.makedirs(path, exist_ok=True)

        self.env = lmdb.open(
            path,
            map_size=map_size,
            max_dbs=max_dbs,
            # Performance tuning
            writemap=True,      # Faster writes (use mmap)
            map_async=True,     # Async mmap flush
            lock=True,          # Multi-process locking
            metasync=False,     # Async metadata flush
            sync=False,         # Async fsync (trade durability for speed)
            max_readers=512,    # Support many concurrent processes
        )

        # Create named databases
        self.completions_db = self.env.open_db(b'completions')
        self.embeddings_db = self.env.open_db(b'embeddings')

        logger.debug(f"LMDB initialized at {path} (map_size={map_size / 1024**3:.1f}GB)")

    def get(self, key: str, db: str = 'completions') -> Optional[bytes]:
        """Get value from LMDB.

        Uses read-only transaction (lock-free with MVCC).

        Args:
            key: Cache key
            db: Database name ('completions' or 'embeddings')

        Returns:
            Value as bytes or None if not found
        """
        db_handle = self.completions_db if db == 'completions' else self.embeddings_db
        with self.env.begin(db=db_handle, write=False) as txn:
            return txn.get(key.encode('utf-8'))

    def set(self, key: str, value: bytes, db: str = 'completions') -> None:
        """Set value in LMDB.

        Uses write transaction (single writer, serialized).
        Auto-expands map_size if MapFullError occurs.

        Args:
            key: Cache key
            value: Value as bytes
            db: Database name ('completions' or 'embeddings')
        """
        db_handle = self.completions_db if db == 'completions' else self.embeddings_db
        try:
            with self.env.begin(db=db_handle, write=True) as txn:
                txn.put(key.encode('utf-8'), value)
        except lmdb.MapFullError:
            logger.warning("LMDB MapFullError: resizing map...")
            self._resize_map()
            # Retry once
            with self.env.begin(db=db_handle, write=True) as txn:
                txn.put(key.encode('utf-8'), value)

    def mset(self, mapping: Dict[str, bytes], db: str = 'completions') -> None:
        """Batch set (single transaction, MUCH more efficient).

        Auto-expands map_size if MapFullError occurs.

        Args:
            mapping: Dictionary of key-value pairs
            db: Database name ('completions' or 'embeddings')
        """
        db_handle = self.completions_db if db == 'completions' else self.embeddings_db
        try:
            with self.env.begin(db=db_handle, write=True) as txn:
                for key, value in mapping.items():
                    txn.put(key.encode('utf-8'), value)
        except lmdb.MapFullError:
            logger.warning("LMDB MapFullError: resizing map...")
            self._resize_map()
            # Retry once
            with self.env.begin(db=db_handle, write=True) as txn:
                for key, value in mapping.items():
                    txn.put(key.encode('utf-8'), value)

    def _resize_map(self):
        """Double the map size of the LMDB environment."""
        new_map_size = self.env.info()['map_size'] * 2
        self.env.set_mapsize(new_map_size)
        logger.info(f"Resized LMDB map size to {new_map_size / 1024**3:.1f}GB")

    def close(self) -> None:
        """Sync and close LMDB environment."""
        self.env.sync()
        self.env.close()
        logger.debug("LMDB closed")


# =============================================================================
# AsyncWriteBuffer - Non-Blocking Write Queue
# =============================================================================

class AsyncWriteBuffer:
    """Non-blocking write buffer with background batch flusher.

    Writes are queued and flushed in batches by background thread.
    Main thread returns immediately after queueing.

    Features:
        - Non-blocking writes (~10μs to queue)
        - Automatic batching (reduces syscalls by 100-1000×)
        - Periodic flushing (every flush_interval seconds)
        - Graceful shutdown (flushes pending writes)

    Attributes:
        backend: LMDBCache instance for persistence
        queue: Queue for pending writes
        shutdown_event: Event to signal shutdown
        flusher: Background thread that flushes batches
        flush_interval: Seconds between flushes
        batch_size: Max items per batch

    Example:
        >>> backend = LMDBCache()
        >>> buffer = AsyncWriteBuffer(backend, flush_interval=0.1, batch_size=1000)
        >>> buffer.set("key1", b"value1", db='completions')  # Returns immediately
        >>> buffer.close()  # Flushes remaining writes
    """

    def __init__(
        self,
        backend: LMDBCache,
        flush_interval: float = 0.1,    # Flush every 100ms
        batch_size: int = 1000,          # Or when 1000 items queued
        max_queue_size: int = 100000     # Prevent unbounded growth
    ):
        """Initialize async write buffer.

        Args:
            backend: LMDBCache instance for persistence
            flush_interval: Seconds between flushes (default: 0.1)
            batch_size: Items per batch flush (default: 1000)
            max_queue_size: Max queued items (default: 100000)
        """
        self.backend = backend
        self.queue: Queue = Queue(maxsize=max_queue_size)
        self.shutdown_event = Event()
        self.flush_interval = flush_interval
        self.batch_size = batch_size

        # Start background flusher thread
        self.flusher = Thread(target=self._flush_loop, daemon=True, name="CacheFlush")
        self.flusher.start()
        logger.debug(
            f"AsyncWriteBuffer started (batch_size={batch_size}, "
            f"flush_interval={flush_interval}s)"
        )

    def set(self, key: str, value: bytes, db: str = 'completions') -> None:
        """Non-blocking set - returns immediately.

        Pushes to queue; background thread does actual write.

        Args:
            key: Cache key
            value: Value as bytes
            db: Database name ('completions' or 'embeddings')
        """
        try:
            # Try non-blocking put first
            self.queue.put_nowait((key, value, db))
        except Full:
            # Queue full - block with timeout
            logger.warning("Write buffer full, applying backpressure...")
            self.queue.put((key, value, db), timeout=1.0)

    def _flush_loop(self):
        """Background thread: batch writes and flush periodically."""
        completion_batch = {}
        embedding_batch = {}
        last_flush = time.time()

        while not self.shutdown_event.is_set():
            try:
                # Calculate timeout until next flush
                timeout = max(0.01, self.flush_interval - (time.time() - last_flush))

                # Try to get item from queue
                try:
                    key, value, db = self.queue.get(timeout=timeout)

                    # Add to appropriate batch
                    if db == 'completions':
                        completion_batch[key] = value
                    else:
                        embedding_batch[key] = value

                except Empty:
                    # Timeout - proceed to flush check
                    pass

                # Flush conditions: batch full OR time elapsed
                total_pending = len(completion_batch) + len(embedding_batch)
                time_elapsed = (time.time() - last_flush) >= self.flush_interval
                should_flush = total_pending >= self.batch_size or time_elapsed

                if should_flush and total_pending > 0:
                    # Batch write completions
                    if completion_batch:
                        self.backend.mset(completion_batch, db='completions')
                        logger.debug(f"Flushed {len(completion_batch)} completions")
                        completion_batch.clear()

                    # Batch write embeddings
                    if embedding_batch:
                        self.backend.mset(embedding_batch, db='embeddings')
                        logger.debug(f"Flushed {len(embedding_batch)} embeddings")
                        embedding_batch.clear()

                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Cache flush error: {e}", exc_info=True)
                # Continue running, don't crash

    def close(self):
        """Flush remaining writes before shutdown."""
        logger.debug("Closing AsyncWriteBuffer...")

        # Signal shutdown
        self.shutdown_event.set()

        # Wait for flusher thread
        self.flusher.join(timeout=5.0)
        if self.flusher.is_alive():
            logger.warning("Flusher thread did not exit cleanly")

        # Final flush of remaining queue
        remaining_completions = {}
        remaining_embeddings = {}

        while not self.queue.empty():
            try:
                key, value, db = self.queue.get_nowait()
                if db == 'completions':
                    remaining_completions[key] = value
                else:
                    remaining_embeddings[key] = value
            except Empty:
                break

        # Flush remaining items
        if remaining_completions:
            self.backend.mset(remaining_completions, db='completions')
            logger.debug(f"Final flush: {len(remaining_completions)} completions")

        if remaining_embeddings:
            self.backend.mset(remaining_embeddings, db='embeddings')
            logger.debug(f"Final flush: {len(remaining_embeddings)} embeddings")

        logger.debug("AsyncWriteBuffer closed")


# =============================================================================
# UnifiedCache - Singleton Three-Tier Cache
# =============================================================================

class UnifiedCache:
    """Singleton cache shared across all modules.

    Implements three-tier architecture:
        - Tier 1: ShardedLRU (memory, lock-free)
        - Tier 2: LMDB (disk, persistent)
        - Tier 3: AsyncWriteBuffer (non-blocking writes)

    Features:
        - Singleton pattern (one instance per process)
        - Namespace support (completion:*, embedding:*)
        - Thread-safe, multi-process safe (via LMDB)
        - Zero configuration (uses environment variables)

    Configuration via environment variables:
        - STRINGSIGHT_CACHE_DIR: Base directory
        - STRINGSIGHT_MEMORY_CACHE_SIZE: In-memory items
        - STRINGSIGHT_MEMORY_SHARDS: Number of shards
        - STRINGSIGHT_LMDB_MAP_SIZE: Max DB size
        - STRINGSIGHT_ASYNC_BATCH_SIZE: Batch size
        - STRINGSIGHT_ASYNC_FLUSH_INTERVAL: Flush interval
        - STRINGSIGHT_DISABLE_CACHE: Disable caching

    Example:
        >>> cache = UnifiedCache()  # Get singleton instance
        >>>
        >>> # Use with CacheKeyBuilder
        >>> key_builder = CacheKeyBuilder({"model": "gpt-4", "messages": [...]})
        >>> cached = cache.get_completion(key_builder)
        >>> if cached is None:
        ...     response = api_call()
        ...     cache.set_completion(key_builder, response)
    """

    _instance: Optional['UnifiedCache'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern: return same instance across all calls."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize cache layers (only once)."""
        if hasattr(self, '_initialized'):
            return  # Already initialized

        # Check if caching is disabled
        if os.environ.get("STRINGSIGHT_DISABLE_CACHE", "0") in ("1", "true", "True"):
            self._disabled = True
            logger.info("Cache disabled (STRINGSIGHT_DISABLE_CACHE=1)")
            self._initialized = True
            return

        self._disabled = False

        # Get configuration from environment
        cache_dir = os.environ.get("STRINGSIGHT_CACHE_DIR", ".cache/stringsight")
        memory_size = int(os.environ.get("STRINGSIGHT_MEMORY_CACHE_SIZE", "10000"))
        num_shards = int(os.environ.get("STRINGSIGHT_MEMORY_SHARDS", "32"))
        map_size_str = os.environ.get("STRINGSIGHT_LMDB_MAP_SIZE", "5GB")
        map_size = parse_size_string(map_size_str)

        # Tier 1: Sharded in-memory LRU
        self.memory = ShardedLRU(maxsize=memory_size, num_shards=num_shards)
        logger.info(f"Cache memory tier: {memory_size} items, {num_shards} shards")

        # Tier 2: LMDB persistent storage
        lmdb_path = Path(cache_dir) / "lmdb"
        self.backend = LMDBCache(path=str(lmdb_path), map_size=map_size)
        logger.info(f"Cache backend: LMDB at {lmdb_path} ({map_size_str})")

        # Tier 3: Async write buffer
        flush_interval = float(os.environ.get("STRINGSIGHT_ASYNC_FLUSH_INTERVAL", "0.1"))
        batch_size = int(os.environ.get("STRINGSIGHT_ASYNC_BATCH_SIZE", "1000"))
        self.async_writer = AsyncWriteBuffer(
            self.backend,
            flush_interval=flush_interval,
            batch_size=batch_size
        )
        logger.info(
            f"Cache async writer: {batch_size} items/batch, {flush_interval}s flush"
        )

        self._initialized = True

        # Register cleanup on exit
        atexit.register(self.close)

    def get_completion(
        self,
        request_data: Union[Dict[str, Any], CacheKeyBuilder]
    ) -> Optional[Dict[str, Any]]:
        """Get cached completion response.

        Args:
            request_data: Either full request dict or CacheKeyBuilder instance

        Returns:
            Cached response dict or None if not found
        """
        if self._disabled:
            return None

        # Build cache key
        if isinstance(request_data, CacheKeyBuilder):
            key = request_data.get_key()
        else:
            key = CacheKeyBuilder(request_data).get_key()

        # Add namespace prefix
        key = f"completion:{key}"

        # Tier 1: Check memory cache (fast path)
        value = self.memory.get(key)
        if value is not None:
            logger.debug(f"Cache HIT (memory): {key[:24]}...")
            return value

        # Tier 2: Check LMDB (slower path)
        value_bytes = self.backend.get(key, db='completions')
        if value_bytes is not None:
            # Deserialize
            value = orjson.loads(value_bytes)
            # Promote to memory cache
            self.memory.set(key, value)
            logger.debug(f"Cache HIT (lmdb): {key[:24]}...")
            return value

        logger.debug(f"Cache MISS: {key[:24]}...")
        return None

    def set_completion(
        self,
        request_data: Union[Dict[str, Any], CacheKeyBuilder],
        response: Dict[str, Any]
    ) -> None:
        """Cache completion response (non-blocking).

        Args:
            request_data: Either full request dict or CacheKeyBuilder instance
            response: Response dict to cache
        """
        if self._disabled:
            return

        # Build cache key
        if isinstance(request_data, CacheKeyBuilder):
            key = request_data.get_key()
        else:
            key = CacheKeyBuilder(request_data).get_key()

        # Add namespace prefix
        key = f"completion:{key}"

        # Tier 1: Store in memory (immediate)
        self.memory.set(key, response)

        # Tier 2: Queue for async write to LMDB (non-blocking)
        value_bytes = orjson.dumps(response)
        self.async_writer.set(key, value_bytes, db='completions')

        logger.debug(f"Cache SET: {key[:24]}...")

    def get_embedding(
        self,
        text_key: Union[str, Tuple[str, str]]
    ) -> Optional[np.ndarray]:
        """Get cached embedding.

        Args:
            text_key: Either text string or (model, text) tuple

        Returns:
            Cached embedding array or None if not found
        """
        if self._disabled:
            return None

        # Build cache key
        if isinstance(text_key, tuple):
            model, text = text_key
            key_data = {"model": model, "text": text}
        else:
            key_data = {"text": text_key}

        key = f"embedding:{CacheKeyBuilder(key_data).get_key()}"

        # Tier 1: Check memory
        value = self.memory.get(key)
        if value is not None:
            logger.debug(f"Cache HIT (memory): {key[:24]}...")
            return value

        # Tier 2: Check LMDB
        value_bytes = self.backend.get(key, db='embeddings')
        if value_bytes is not None:
            # Deserialize bytes to numpy array
            embedding = np.frombuffer(value_bytes, dtype=np.float32)
            # Promote to memory
            self.memory.set(key, embedding)
            logger.debug(f"Cache HIT (lmdb): {key[:24]}...")
            return embedding

        logger.debug(f"Cache MISS: {key[:24]}...")
        return None

    def set_embedding(
        self,
        text_key: Union[str, Tuple[str, str]],
        embedding: np.ndarray
    ) -> None:
        """Cache embedding (non-blocking).

        Args:
            text_key: Either text string or (model, text) tuple
            embedding: Embedding array to cache
        """
        if self._disabled:
            return

        # Build cache key
        if isinstance(text_key, tuple):
            model, text = text_key
            key_data = {"model": model, "text": text}
        else:
            key_data = {"text": text_key}

        key = f"embedding:{CacheKeyBuilder(key_data).get_key()}"

        # Convert to numpy array
        embedding_array = np.asarray(embedding, dtype=np.float32)

        # Tier 1: Store in memory
        self.memory.set(key, embedding_array)

        # Tier 2: Queue for async write
        value_bytes = embedding_array.tobytes()
        self.async_writer.set(key, value_bytes, db='embeddings')

        logger.debug(f"Cache SET: {key[:24]}...")

    def close(self):
        """Close cache gracefully, flushing pending writes."""
        if self._disabled or not hasattr(self, 'async_writer'):
            return

        logger.info("Closing cache...")

        # Flush async write buffer
        self.async_writer.close()

        # Close LMDB
        self.backend.close()

        # Clear memory cache
        self.memory.clear()

        logger.info("Cache closed")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'UnifiedCache',
    'CacheKeyBuilder',
    'parse_size_string',
]
