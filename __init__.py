"""zyf-WAN-Lora-Loader

A ComfyUI custom node for WAN 2.2 that loads _high/_low paired LoRAs
in a single node — drop in a high_noise MODEL and a low_noise MODEL,
configure your LoRA stack, and pipe the patched pair back out.

Inspired by rgthree-comfy's Power Lora Loader.
"""

from .wan_lora_loader import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
