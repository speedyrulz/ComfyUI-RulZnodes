"""Backend for the RulZ Image Browser sidebar tab (web/image_browser.js).

Three routes: list a directory, serve a JPEG thumbnail of one image, and copy the
picked images into ComfyUI's input folder so RulZ Load Image Batch (HEIC) can load
them the same way it loads dropped files. Only image files are ever read or copied.
"""
import asyncio
import hashlib
import io
import os
import shutil
import string
import threading
from collections import OrderedDict

from aiohttp import web
from PIL import Image, ImageOps

import folder_paths
from server import PromptServer

from .image_utils import to_float_rgb
from .load_image_batch import _natural_key
from .load_image_heic import HEIF_AVAILABLE, HEIF_EXTENSIONS, IMAGE_EXTENSIONS

THUMB_SIZE = 384
_THUMB_CACHE_MAX = 512
_thumb_cache = OrderedDict()  # (path, mtime) -> jpeg bytes
_thumb_lock = threading.Lock()

routes = PromptServer.instance.routes


def _extensions():
    return IMAGE_EXTENSIONS | (HEIF_EXTENSIONS if HEIF_AVAILABLE else set())


def _is_image(name):
    return os.path.splitext(name)[1].lower() in _extensions()


def _clean_path(raw):
    """Path as typed -> absolute. Accepts ~ and the quotes Windows "Copy as path" adds."""
    raw = (raw or "").strip().strip("\"'").strip()
    if not raw:
        return ""
    return os.path.abspath(os.path.expanduser(raw))


def _roots():
    if os.name == "nt":
        drives = [f"{letter}:\\" for letter in string.ascii_uppercase]
        return [d for d in drives if os.path.exists(d)]
    return ["/"]


def _scan(path):
    """Listing for the pane. An empty path lists the drives (Windows) or "/"."""
    if not path:
        if os.name != "nt":
            path = "/"
        else:
            return {
                "path": "",
                "parent": None,
                "dirs": [{"name": d, "path": d} for d in _roots()],
                "images": [],
            }

    dirs, images = [], []
    with os.scandir(path) as entries:
        for entry in entries:
            name = entry.name
            # Dot folders and Windows system folders ($RECYCLE.BIN, ...).
            if name.startswith((".", "$")):
                continue
            try:
                if entry.is_dir():
                    dirs.append({"name": name, "path": entry.path})
                elif _is_image(name) and entry.is_file():
                    stat = entry.stat()
                    images.append(
                        {
                            "name": name,
                            "path": entry.path,
                            "mtime": int(stat.st_mtime),
                            "size": stat.st_size,
                        }
                    )
            except OSError:
                continue  # unreadable entry (broken link, no permission)

    dirs.sort(key=lambda d: _natural_key(d["name"]))
    images.sort(key=lambda i: _natural_key(i["name"]))

    parent = os.path.dirname(path)
    if parent == path:  # a drive root or "/"
        parent = "" if os.name == "nt" else None
    return {"path": path, "parent": parent, "dirs": dirs, "images": images}


def _make_thumb(path):
    key = (path, os.path.getmtime(path))
    with _thumb_lock:
        data = _thumb_cache.get(key)
        if data is not None:
            _thumb_cache.move_to_end(key)
            return data

    with Image.open(path) as img:
        # JPEGs can decode straight to a smaller size, which is much faster.
        img.draft("RGB", (THUMB_SIZE, THUMB_SIZE))
        img = ImageOps.exif_transpose(img)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        if img.mode.startswith("I"):  # 16-bit greyscale; convert("RGB") would clip it
            img = Image.fromarray((to_float_rgb(img) * 255.0).astype("uint8"))
        else:
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
    data = buf.getvalue()

    with _thumb_lock:
        _thumb_cache[key] = data
        while len(_thumb_cache) > _THUMB_CACHE_MAX:
            _thumb_cache.popitem(last=False)
    return data


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def _same_content(a, b):
    return os.path.getsize(a) == os.path.getsize(b) and _sha256(a) == _sha256(b)


def _import_one(src, input_dir):
    """Make `src` loadable by the Load Image nodes; returns its name relative to the
    input folder. Mirrors ComfyUI's upload: a file that is already there with the
    same content is reused, a different file with the same name gets " (n)"."""
    src = _clean_path(src)
    if not src or not _is_image(src) or not os.path.isfile(src):
        raise FileNotFoundError(f"not an image file: {src}")

    try:
        inside = os.path.commonpath([input_dir, src]) == input_dir
    except ValueError:  # different drives
        inside = False
    if inside:
        return os.path.relpath(src, input_dir).replace(os.sep, "/")

    stem, ext = os.path.splitext(os.path.basename(src))
    dest = os.path.join(input_dir, stem + ext)
    n = 1
    while os.path.exists(dest):
        if _same_content(src, dest):
            return os.path.basename(dest)
        dest = os.path.join(input_dir, f"{stem} ({n}){ext}")
        n += 1
    shutil.copy2(src, dest)
    return os.path.basename(dest)


def _import_all(paths):
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    os.makedirs(input_dir, exist_ok=True)
    files, errors = [], []
    for src in paths:
        try:
            files.append(_import_one(src, input_dir))
        except Exception as e:
            errors.append(f"{os.path.basename(str(src))}: {e}")
    return {"files": files, "errors": errors}


def _error(status, message):
    return web.json_response({"error": message}, status=status)


@routes.get("/rulz/image_browser/list")
async def list_directory(request):
    path = _clean_path(request.rel_url.query.get("path", ""))
    if path and not os.path.isdir(path):
        return _error(404, f"Folder not found: {path}")
    try:
        loop = asyncio.get_running_loop()
        return web.json_response(await loop.run_in_executor(None, _scan, path))
    except PermissionError:
        return _error(403, f"No permission to read: {path}")
    except OSError as e:
        return _error(500, f"Could not read {path}: {e}")


@routes.get("/rulz/image_browser/thumb")
async def thumbnail(request):
    path = _clean_path(request.rel_url.query.get("path", ""))
    if not path or not _is_image(path) or not os.path.isfile(path):
        return web.Response(status=404)
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _make_thumb, path)
    except Exception as e:
        print(f"[RulZ Image Browser] thumbnail failed for {path}: {e}")
        return web.Response(status=500)
    # The pane puts the file's mtime in the URL, so a changed file is a new URL.
    return web.Response(
        body=data,
        content_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@routes.post("/rulz/image_browser/import")
async def import_images(request):
    try:
        paths = (await request.json()).get("paths")
    except Exception:
        return _error(400, "Invalid request body")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        return _error(400, "'paths' must be a list of file paths")
    loop = asyncio.get_running_loop()
    return web.json_response(await loop.run_in_executor(None, _import_all, paths))
