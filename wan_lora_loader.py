"""Wan 2.2 Paired LoRA Loader.

A single-node solution for loading multiple _high / _low paired LoRAs onto
the high_noise and low_noise diffusion models in one shot.

Inspired by rgthree-comfy's Power Lora Loader.
"""

from __future__ import annotations

import os
import re
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import folder_paths

# `nodes.LoraLoader` and `server.PromptServer` are imported lazily below so
# the module can be parsed in environments that don't have `torch` (e.g. CI
# or unit tests where we only need the pairing helpers).

NODE_NAME = "WanLoraLoader (zyf-WAN-Lora-Loader)"
DISPLAY_NAME = "Wan 2.2 LoRA Loader (zyf)"
SINGLE_NODE_NAME = "SingleLoraLoader (zyf-WAN-Lora-Loader)"
SINGLE_DISPLAY_NAME = "Simple LoRA Loader (zyf)"
CATEGORY = "zyf-WAN-Lora-Loader"

# Patterns that match a "_high"/"_low" / "_HIGH"/"_LOW" marker in a WAN 2.2
# paired LoRA filename.  The marker can be:
#   * at the end of the stem:        "foo_high"
#   * followed by a separator:       "foo_high_playtime"  /  "foo_HIGH.suffix"
#
# The regex intentionally doesn't consume the separator BEFORE the marker
# (it stays in the base name); the lookahead only constrains the character
# AFTER the marker, so the trailing separator is captured and stripped.
_HIGH_LOW_RE = re.compile(
    r"(?P<side>high|HIGH|High|low|LOW|Low)(?P<tail>[\._\-]|$)"
)
_LORA_EXTS = (".safetensors", ".pt", ".ckpt", ".bin")


def _strip_ext(name: str) -> str:
    for ext in _LORA_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def _split_pair_basename(filename: str) -> Optional[Tuple[str, str]]:
    """Return ("high"|"low", base_name) if filename looks like a paired file.

    Examples:
        "doggy_style_high_playtime.safetensors" -> ("high", "doggy_style_playtime")
        "iGoon_Blink_Nude_Posing_12V_HIGH.safetensors" -> ("high", "iGoon_Blink_Nude_Posing_12V")
        "foo_low.safetensors" -> ("low", "foo")
    """
    base = _strip_ext(filename)
    m = _HIGH_LOW_RE.search(base)
    if not m:
        return None
    side = m.group("side").lower()
    # Drop the marker (and the optional separator after it) to get the base
    # name.  If the marker was at the end of the stem, ``tail`` is an empty
    # string; otherwise it's a separator character we also strip out.
    base_name = base[:m.start()] + base[m.end():]
    base_name = re.sub(r"^[\._\-]+|[\._\-]+$", "", base_name)
    if not base_name:
        return None
    return side, base_name


def list_paired_loras() -> List[Dict[str, Any]]:
    """Scan the configured `loras` folder and return paired LoRAs.

    A pair is two files whose stems (sans _high/_low) match exactly.
    Returns a list of dicts:
        {
          "name": "<base name>",
          "high": "<filename>" or None,
          "low": "<filename" or None,
        }
    Sorted by `name` case-insensitively.
    """
    try:
        lora_files = folder_paths.get_filename_list("loras")
    except Exception:  # pragma: no cover - folder_paths not yet ready
        return []

    # Map case-folded base_name -> {"high": ..., "low": ...}
    grouped: Dict[str, Dict[str, Optional[str]]] = {}
    for filename in lora_files:
        info = _split_pair_basename(filename)
        if info is None:
            continue
        side, base = info
        key = base.lower()
        entry = grouped.setdefault(key, {"high": None, "low": None, "name": base})
        if entry[side] is None:
            entry[side] = filename
        # Keep the prettiest base name (mix of cases – prefer the one that
        # has the most lowercase matches; just take the first one we see).
        if not entry["name"] or entry["name"].lower() == base.lower() and base != entry["name"]:
            # Update name to preserve original case if different.
            if entry["name"] is None or entry["name"].lower() == base.lower():
                if base != entry["name"]:
                    entry["name"] = base

    pairs: List[Dict[str, Any]] = []
    for entry in grouped.values():
        if entry["high"] or entry["low"]:
            pairs.append(entry)
    pairs.sort(key=lambda e: e["name"].lower())
    return pairs


def list_all_loras() -> List[str]:
    """Return all available LoRA filenames (no pairing)."""
    try:
        return list(folder_paths.get_filename_list("loras"))
    except Exception:  # pragma: no cover
        return []


def find_pair_for(lora_filename: str, side: str = "auto") -> Dict[str, Any]:
    """Given one LoRA filename, find its counterpart in the same directory.

    The user stores each high/low pair in its own subfolder.  We list all
    LoRA files in that directory, and return them all as `siblings`.

    When `side` is "high" or "low", the selected file is assigned to that
    side and the other file (if any) goes to the opposite side.
    When `side` is "auto" (default), files are assigned alphabetically
    (first -> high, second -> low).

    Returns:
        {
          "selected": "<filename>",
          "lora_high": "<filename>" | None,
          "lora_low": "<filename>" | None,
          "display_name": "<folder name>",
          "siblings": ["<file1>", "<file2>", ...],
        }
    """
    if not lora_filename:
        return {}

    # Normalize path separators so comparisons work regardless of whether
    # the frontend sends "/" or the backend stores "\".
    lora_filename = lora_filename.replace("\\", "/")
    dirname = os.path.dirname(lora_filename)
    filename_basename = os.path.basename(lora_filename)

    # List all LoRA files in the same directory.
    try:
        all_files = folder_paths.get_filename_list("loras")
    except Exception:
        all_files = []

    sibling_files = []
    for f in all_files:
        f_norm = f.replace("\\", "/")
        f_dirname = os.path.dirname(f_norm)
        if f_dirname == dirname:
            f_base = os.path.basename(f_norm)
            if f_base.lower().endswith(_LORA_EXTS):
                sibling_files.append(f_norm)

    # Sort alphabetically by filename (case-insensitive).
    sibling_files.sort(key=lambda x: os.path.basename(x).lower())

    # Determine high/low assignment based on side.
    other = [f for f in sibling_files if f != lora_filename]
    if side == "high":
        lora_high = lora_filename
        lora_low = other[0] if other else None
    elif side == "low":
        lora_low = lora_filename
        lora_high = other[0] if other else None
    else:
        # auto: alphabetical order (first -> high, second -> low)
        lora_high = sibling_files[0] if len(sibling_files) >= 1 else None
        lora_low = sibling_files[1] if len(sibling_files) >= 2 else None

    # Display name = the folder name.
    display_name = os.path.basename(dirname) if dirname else _strip_ext(filename_basename)

    return {
        "selected": lora_filename,
        "lora_high": lora_high,
        "lora_low": lora_low,
        "display_name": display_name,
        "siblings": sibling_files,
    }


def build_lora_tree() -> List[Dict[str, Any]]:
    """Build a nested folder tree of all LoRA files.

    Returns a list of nodes, each being either:
      { "type": "folder", "name": "<dirname>", "children": [...] }
      { "type": "file",   "name": "<filename>", "path": "<relative_path>" }

    Folders are sorted alphabetically first, then files.
    """
    try:
        lora_dir = folder_paths.get_folder_paths("loras")
        if not lora_dir:
            return []
        base_dir = lora_dir[0]
    except Exception:
        return []

    if not os.path.isdir(base_dir):
        return []

    # Walk the directory tree to collect all folders and files.
    parent_map: Dict[str, List[tuple]] = {}

    for root, dirs, files in os.walk(base_dir):
        # Get relative path from base_dir.
        rel_root = os.path.relpath(root, base_dir)
        if rel_root == ".":
            rel_root = "."
        else:
            # Normalize to forward slashes for consistency.
            rel_root = rel_root.replace("\\", "/")

        # Add this directory to parent_map.
        parent = os.path.dirname(rel_root) or "."
        if rel_root != ".":
            dir_name = os.path.basename(rel_root)
            parent_map.setdefault(parent, []).append((dir_name, True, rel_root))

        # Add LoRA files in this directory.
        for f in files:
            if f.lower().endswith(_LORA_EXTS):
                file_rel = f"{rel_root}/{f}" if rel_root != "." else f
                parent_map.setdefault(rel_root, []).append((f, False, file_rel))

        # Sort dirs in-place to ensure consistent ordering.
        dirs.sort()

    def _natural_sort_key(s: str):
        """Split string into alternating text/number parts for natural sort.

        E.g. "09_foo" -> ["09", "_foo"] -> [9, "_foo"]
             "100_bar" -> ["100", "_bar"] -> [100, "_bar"]
        This ensures "100" sorts after "99" instead of after "09".
        """
        parts = re.split(r"(\d+)", s)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    def _build_from(parent: str) -> List[Dict[str, Any]]:
        children = parent_map.get(parent, [])
        # Sort: folders first (natural sort), then files (natural sort).
        folders = sorted(
            [c for c in children if c[1]],
            key=lambda c: _natural_sort_key(c[0]),
        )
        files = sorted(
            [c for c in children if not c[1]],
            key=lambda c: _natural_sort_key(c[0]),
        )
        nodes: List[Dict[str, Any]] = []
        for name, _, path in folders:
            nodes.append({
                "type": "folder",
                "name": name,
                "children": _build_from(path),
            })
        for name, _, path in files:
            nodes.append({
                "type": "file",
                "name": name,
                "path": path,
            })
        return nodes

    return _build_from(".")


# ---------------------------------------------------------------------------
# Server routes
# ---------------------------------------------------------------------------

def _read_txt_file(path: str) -> str:
    """Read a text file, trying multiple encodings to avoid garbled output.

    On Chinese Windows systems, TXT files are often encoded in GBK / GB2312
    rather than UTF-8.  We try the most common encodings in order.
    """
    # Read raw bytes first so we can retry without re-opening.
    with open(path, "rb") as f:
        raw = f.read()

    # BOM detection: UTF-8 BOM → strip and decode as UTF-8
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    # UTF-16 LE BOM
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace")
    # UTF-16 BE BOM
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")

    # Try encodings in order of likelihood.
    for enc in ("utf-8", "gbk", "gb2312", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    # Ultimate fallback – never fails but may produce mojibake.
    return raw.decode("latin-1")


def find_txt_for(lora_filename: str) -> Dict[str, Any]:
    """Find a .txt file in the same directory as the given LoRA file.

    Returns:
        { "found": True, "path": "<relative_path>", "content": "...", "encoding": "..." }
        or { "found": False }
    """
    if not lora_filename:
        return {"found": False}

    lora_filename = lora_filename.replace("\\", "/")
    dirname = os.path.dirname(lora_filename)

    try:
        lora_dir = folder_paths.get_folder_paths("loras")
        if not lora_dir:
            return {"found": False}
        base_dir = lora_dir[0]
    except Exception:
        return {"found": False}

    abs_dir = os.path.join(base_dir, dirname.replace("/", os.sep))
    if not os.path.isdir(abs_dir):
        return {"found": False}

    # Find the first .txt file in the directory.
    txt_files = [f for f in os.listdir(abs_dir) if f.lower().endswith(".txt")]
    if not txt_files:
        return {"found": False}

    txt_files.sort(key=lambda x: x.lower())
    txt_name = txt_files[0]
    txt_rel = f"{dirname}/{txt_name}" if dirname else txt_name
    txt_path = os.path.join(abs_dir, txt_name)

    try:
        content, encoding = _read_txt_file_with_encoding(txt_path)
    except Exception:
        content = ""
        encoding = "utf-8"

    return {"found": True, "path": txt_rel, "content": content, "encoding": encoding}


def _read_txt_file_with_encoding(path: str) -> Tuple[str, str]:
    """Read a text file, trying multiple encodings. Returns (content, encoding_used)."""
    with open(path, "rb") as f:
        raw = f.read()

    # BOM detection
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8"
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace"), "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace"), "utf-16-be"

    for enc in ("utf-8", "gbk", "gb2312", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue

    return raw.decode("latin-1"), "latin-1"


def save_txt_file(txt_rel_path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Save content back to the TXT file at the given relative path.

    Always writes as UTF-8 WITH BOM (``utf-8-sig``) regardless of the
    detected original encoding, for maximum cross-platform compatibility:

    * Chinese Windows 资源管理器 reads the BOM and opens the file as UTF-8
      instead of guessing GBK and producing mojibake (乱码).
    * Mobile apps can identify the file as a valid TXT.
    * The plugin reader's BOM detection handles it on the next open.

    The ``encoding`` argument is kept for API compatibility but ignored —
    the on-disk format is always UTF-8 with BOM.

    Returns:
        { "success": True, "encoding": "utf-8-sig" }
        or { "success": False, "error": "..." }
    """
    if not txt_rel_path:
        return {"success": False, "error": "No path provided"}

    txt_rel_path = txt_rel_path.replace("\\", "/")
    try:
        lora_dir = folder_paths.get_folder_paths("loras")
        if not lora_dir:
            return {"success": False, "error": "LoRA folder not found"}
        base_dir = lora_dir[0]
    except Exception as e:
        return {"success": False, "error": str(e)}

    abs_path = os.path.join(base_dir, txt_rel_path.replace("/", os.sep))
    abs_dir = os.path.dirname(abs_path)
    if not os.path.isdir(abs_dir):
        return {"success": False, "error": f"Directory not found: {abs_dir}"}

    try:
        # Always write UTF-8 with BOM ("utf-8-sig"). This prepends the
        # 0xEF 0xBB 0xBF signature so Windows and mobile apps can detect
        # the encoding correctly.  We also strip any leading BOM the editor
        # may have already injected into `content` to avoid a double BOM.
        if content.startswith("\ufeff"):
            content = content[1:]
        with open(abs_path, "w", encoding="utf-8-sig") as f:
            f.write(content)
        return {"success": True, "encoding": "utf-8-sig"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def open_lora_folder(lora_filename: str) -> Dict[str, Any]:
    """Open the folder containing the LoRA file in the system file manager.

    * On Windows, ``explorer /select,<path>`` reveals the file inside
      Explorer (the user sees the file highlighted).
    * On macOS, ``open -R <path>`` reveals the file inside Finder.
    * On Linux, ``xdg-open <dir>`` opens the containing directory.

    Returns:
        { "success": True } or { "success": False, "error": "..." }
    """
    if not lora_filename:
        return {"success": False, "error": "No path provided"}

    lora_filename = lora_filename.replace("\\", "/")
    dirname = os.path.dirname(lora_filename)

    try:
        lora_dir = folder_paths.get_folder_paths("loras")
        if not lora_dir:
            return {"success": False, "error": "LoRA folder not found"}
        base_dir = lora_dir[0]
    except Exception as e:
        return {"success": False, "error": str(e)}

    abs_dir = os.path.join(base_dir, dirname.replace("/", os.sep)) if dirname else base_dir
    abs_path = os.path.join(base_dir, lora_filename.replace("/", os.sep))

    if not os.path.isdir(abs_dir):
        return {"success": False, "error": f"Directory not found: {abs_dir}"}
    if not os.path.isfile(abs_path):
        return {"success": False, "error": f"File not found: {abs_path}"}

    try:
        if sys.platform == "win32":
            # ``/select,<path>`` reveals the file inside Explorer.  The
            # comma must be glued to the path (Windows convention).
            subprocess.Popen(["explorer", f"/select,{os.path.normpath(abs_path)}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", abs_path])
        else:
            # Linux / other Unix – just open the containing directory.
            subprocess.Popen(["xdg-open", abs_dir])
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _register_routes() -> None:
    try:
        from server import PromptServer
    except Exception:  # pragma: no cover - server module not importable
        return
    server = PromptServer.instance
    if server is None:  # pragma: no cover - imported too early
        return

    from aiohttp import web

    @server.routes.get("/zyf_wan_lora/lora-tree")
    async def _get_lora_tree(_request):  # noqa: ANN001
        return web.json_response({"tree": build_lora_tree()})

    @server.routes.get("/zyf_wan_lora/find-pair")
    async def _find_pair(request):  # noqa: ANN001
        lora = request.query.get("lora", "")
        side = request.query.get("side", "auto")
        return web.json_response(find_pair_for(lora, side))

    @server.routes.get("/zyf_wan_lora/find-txt")
    async def _find_txt(request):  # noqa: ANN001
        lora = request.query.get("lora", "")
        return web.json_response(find_txt_for(lora))

    @server.routes.post("/zyf_wan_lora/save-txt")
    async def _save_txt(request):  # noqa: ANN001
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"})
        txt_path = body.get("path", "")
        content = body.get("content", "")
        encoding = body.get("encoding", "utf-8")
        return web.json_response(save_txt_file(txt_path, content, encoding))

    @server.routes.get("/zyf_wan_lora/open-folder")
    async def _open_folder(request):  # noqa: ANN001
        lora = request.query.get("lora", "")
        return web.json_response(open_lora_folder(lora))


# Register routes as soon as the module is imported – ComfyUI does this when
# it loads the custom node package.
try:
    _register_routes()
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Node definition
# ---------------------------------------------------------------------------


class FlexibleOptionalInputType(dict):
    """A drop-in clone of rgthree's helper (avoids the dependency)."""

    def __init__(self, type_: Any, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.type = type_
        self.data = data or {}
        for k, v in self.data.items():
            self[k] = v

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):  # type: ignore[override]
        return True


class AnyType(str):
    """Matches any other type in ComfyUI's input type check."""

    def __ne__(self, other: object) -> bool:  # type: ignore[operator]
        return False


ANY = AnyType("*")


class WanLoraLoader:
    """Loads a list of (high, low) paired LoRAs onto two diffusion models."""

    NAME = NODE_NAME
    DISPLAY_NAME = DISPLAY_NAME
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Drop-in paired-LoRA loader for Wan 2.2: takes a high_noise MODEL and a "
        "low_noise MODEL upstream, applies a stack of _high/_low LoRA pairs in "
        "order, and outputs the patched models downstream. Inspired by "
        "rgthree-comfy's Power Lora Loader."
    )

    # Returning two MODELs preserves the high/low split downstream.
    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("HIGH_NOISE_MODEL", "LOW_NOISE_MODEL")
    FUNCTION = "load_loras"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(ANY, {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
            }),
            "hidden": {},
        }

    # ---- Helpers ---------------------------------------------------------

    @staticmethod
    def _coerce_lora_dict(value: Any) -> Optional[Dict[str, Any]]:
        """Coerce a widget payload (dict or JSON string) into our LoRA shape."""
        if value is None:
            return None
        if isinstance(value, str):
            import json
            try:
                value = json.loads(value)
            except Exception:
                return None
        if not isinstance(value, dict):
            return None
        return value

    @staticmethod
    def _apply_lora(model, lora_name: Optional[str], strength: float):
        """Apply a single LoRA to a model.  Returns the patched model.

        Mirrors LoraLoaderModelOnly – we only touch the diffusion model
        because WAN 2.2 doesn't carry CLIP here.
        """
        if model is None or not lora_name or strength == 0:
            return model
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        # Lazy import so ComfyUI's `nodes` module is fully resolved.
        from nodes import LoraLoader  # type: ignore
        loader = LoraLoader()
        model, _clip = loader.load_lora(model, None, lora_name, strength, 0.0)
        return model

    # ---- Main entry ------------------------------------------------------

    def load_loras(self, model_high=None, model_low=None, **kwargs):
        # Walk the lora_<n> inputs in numeric order.  Anything that doesn't
        # decode into our expected dict is silently skipped.
        loras: List[Dict[str, Any]] = []
        for key, value in kwargs.items():
            if not key.lower().startswith("lora_"):
                continue
            data = self._coerce_lora_dict(value)
            if not data:
                continue
            loras.append(data)

        # Sort by the trailing number ("lora_1" -> 1, "lora_10" -> 10).
        def _sort_key(entry: Dict[str, Any]) -> int:
            # We don't have access to the original key here, so fall back to
            # the explicit "index" or 0.
            return int(entry.get("index", 0))

        loras.sort(key=_sort_key)

        for entry in loras:
            if not entry.get("on", True):
                continue
            strength_high = float(entry.get("strength_high", entry.get("strength", 1.0)) or 0.0)
            strength_low = float(entry.get("strength_low", entry.get("strength", 1.0)) or 0.0)
            model_high = self._apply_lora(model_high, entry.get("lora_high"), strength_high)
            model_low = self._apply_lora(model_low, entry.get("lora_low"), strength_low)

        return (model_high, model_low)


class SingleLoraLoader:
    """Loads a stack of LoRAs onto a single MODEL (+ optional CLIP).

    A general-purpose LoRA loader that works with any model/CLIP workflow.
    Unlike the Wan 2.2 loader, there is no high/low noise split and no
    auto-pairing logic — each row selects one LoRA file with a single
    strength value that applies to both MODEL and CLIP.

    The CLIP input is **optional**: when the user does not connect it,
    the loader silently skips the CLIP side and returns ``(model, None)``.
    The MODEL input is still required.
    """

    NAME = SINGLE_NODE_NAME
    DISPLAY_NAME = SINGLE_DISPLAY_NAME
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "General-purpose LoRA loader: takes a MODEL and (optionally) a "
        "CLIP, applies a stack of LoRAs in order, and outputs the patched "
        "MODEL and CLIP. Each row selects one LoRA with a single strength. "
        "No high/low pairing — works with any single-LoRA workflow."
    )

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # ``clip`` is intentionally optional so the node can be used in
        # workflows that only need MODEL-side LoRA application.  When
        # disconnected, the CLIP side is skipped and ``None`` is returned
        # as the CLIP output.
        flexible = FlexibleOptionalInputType(ANY, {})
        flexible["clip"] = ("CLIP",)
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": flexible,
            "hidden": {},
        }

    # ---- Helpers ---------------------------------------------------------

    @staticmethod
    def _coerce_lora_dict(value: Any) -> Optional[Dict[str, Any]]:
        """Coerce a widget payload (dict or JSON string) into our LoRA shape."""
        if value is None:
            return None
        if isinstance(value, str):
            import json
            try:
                value = json.loads(value)
            except Exception:
                return None
        if not isinstance(value, dict):
            return None
        return value

    @staticmethod
    def _apply_lora(model, clip, lora_name: Optional[str], strength: float):
        """Apply a single LoRA to model and (optionally) clip.

        Returns ``(model, clip)``.  When ``clip`` is ``None`` the LoRA
        is only applied to the model and ``None`` is returned for clip.
        """
        if not lora_name or strength == 0:
            return model, clip
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        from nodes import LoraLoader  # type: ignore
        loader = LoraLoader()
        if clip is None:
            # Model-only path – mirror LoraLoaderModelOnly behaviour by
            # setting the clip strength to 0.  The LoraLoader still
            # accepts ``None`` for clip in this mode.
            model, _unused = loader.load_lora(model, None, lora_name, strength, 0.0)
            return model, None
        model, clip = loader.load_lora(model, clip, lora_name, strength, strength)
        return model, clip

    # ---- Main entry ------------------------------------------------------

    def load_loras(self, model, clip=None, **kwargs):
        # Walk the lora_<n> inputs in numeric order.  Anything that doesn't
        # decode into our expected dict is silently skipped.
        loras: List[Dict[str, Any]] = []
        for key, value in kwargs.items():
            if not key.lower().startswith("lora_"):
                continue
            data = self._coerce_lora_dict(value)
            if not data:
                continue
            loras.append(data)

        # Sort by the trailing number ("lora_1" -> 1, "lora_10" -> 10).
        def _sort_key(entry: Dict[str, Any]) -> int:
            return int(entry.get("index", 0))

        loras.sort(key=_sort_key)

        for entry in loras:
            if not entry.get("on", True):
                continue
            strength = float(entry.get("strength", 1.0) or 0.0)
            model, clip = self._apply_lora(
                model, clip, entry.get("lora"), strength,
            )

        return (model, clip)


NODE_CLASS_MAPPINGS = {
    "WanLoraLoader": WanLoraLoader,
    "SingleLoraLoader": SingleLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLoraLoader": DISPLAY_NAME,
    "SingleLoraLoader": SINGLE_DISPLAY_NAME,
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WanLoraLoader",
    "SingleLoraLoader",
]
