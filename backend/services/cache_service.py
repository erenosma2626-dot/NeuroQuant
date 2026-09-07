"""
In-Memory TTL Cache Service
"""

import time
from typing import Any, Optional, Dict

class MemoryCache:
    def __init__(self, default_ttl: int = 900):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        item = self._cache[key]
        if time.time() > item["expires_at"]:
            del self._cache[key]
            return None
        return item["data"]

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self._cache[key] = {
            "data": data,
            "expires_at": time.time() + effective_ttl
        }

    def clear(self) -> None:
        self._cache.clear()

cache = MemoryCache()
