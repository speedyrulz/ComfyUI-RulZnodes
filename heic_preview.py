"""Serve HEIC/HEIF files as JPEG from ComfyUI's /view endpoint.

Browsers can't render HEIC, so the Load Image preview would be blank. This adds an
aiohttp middleware that intercepts /view requests for .heic/.heif files and returns a
JPEG conversion instead. Nothing is written to disk and the original file is untouched.
"""
import asyncio
import io
import os
import threading

from aiohttp import web
from PIL import Image, ImageOps

import folder_paths
from server import PromptServer

HEIF_EXTENSIONS = (".heic", ".heif")
VIEW_PATHS = ("/view", "/api/view")
_cache = {}  # (path, mtime) -> jpeg bytes
_CACHE_MAX = 16
_cache_lock = threading.Lock()  # conversions run in several executor threads at once


def _convert(path):
    key = (path, os.path.getmtime(path))
    with _cache_lock:
        data = _cache.get(key)
    if data is None:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        with _cache_lock:
            while len(_cache) >= _CACHE_MAX:
                _cache.pop(next(iter(_cache)))
            _cache[key] = data
    return data


@web.middleware
async def heic_view_middleware(request, handler):
    if request.method == "GET" and request.path in VIEW_PATHS:
        # aiohttp has already URL-decoded the query (as ComfyUI's /view uses it);
        # decoding again would turn a literal "%20" in a name into a space.
        filename = request.rel_url.query.get("filename", "")
        if filename.lower().endswith(HEIF_EXTENSIONS):
            try:
                subfolder = request.rel_url.query.get("subfolder", "")
                base = folder_paths.get_directory_by_type(
                    request.rel_url.query.get("type", "output")
                )
                if base is not None:
                    base = os.path.abspath(base)
                    path = os.path.abspath(
                        os.path.join(base, subfolder, os.path.basename(filename))
                    )
                    # Stay inside the ComfyUI directory (no path traversal).
                    if path.startswith(base + os.sep) and os.path.isfile(path):
                        loop = asyncio.get_running_loop()
                        data = await loop.run_in_executor(None, _convert, path)
                        return web.Response(body=data, content_type="image/jpeg")
            except Exception as e:
                print(f"[ImageLoadFilenameHEIC] preview conversion failed: {e}")
    return await handler(request)


def install():
    app = PromptServer.instance.app
    if heic_view_middleware not in app.middlewares:
        app.middlewares.append(heic_view_middleware)
