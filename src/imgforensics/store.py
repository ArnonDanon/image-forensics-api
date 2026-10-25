"""Content-addressed image store with a small in-memory cache of decoded images."""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

MAX_BYTES = 40 * 1024 * 1024
MAX_PIXELS = 50_000_000


class ImageStore:
    def __init__(self, root: str | os.PathLike | None = None, cache_size: int = 16):
        self.root = Path(root or os.environ.get("IMGF_DATA_DIR", "data/images"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._size = cache_size
        self._lock = threading.Lock()

    def put(self, data: bytes) -> str:
        if len(data) > MAX_BYTES:
            raise ValueError("image larger than 40 MB")
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("not a decodable image")
        if arr.shape[0] * arr.shape[1] > MAX_PIXELS:
            raise ValueError("image larger than 50 megapixels")
        image_id = hashlib.sha256(data).hexdigest()[:24]
        path = self.root / f"{image_id}.bin"
        if not path.exists():
            path.write_bytes(data)       # keep original bytes untouched
        self._remember(image_id, arr)
        return image_id

    def put_path(self, path: str) -> str:
        return self.put(Path(path).read_bytes())

    def get(self, image_id: str) -> np.ndarray:
        if not image_id.isalnum():
            raise KeyError(image_id)
        with self._lock:
            if image_id in self._cache:
                self._cache.move_to_end(image_id)
                return self._cache[image_id]
        path = self.root / f"{image_id}.bin"
        if not path.exists():
            raise KeyError(image_id)
        arr = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        self._remember(image_id, arr)
        return arr

    def _remember(self, image_id: str, arr: np.ndarray) -> None:
        with self._lock:
            self._cache[image_id] = arr
            self._cache.move_to_end(image_id)
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
