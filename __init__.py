"""ComfyUI KittenTTS 2 Pro — public custom-node entry point.

Model assets and heavyweight dependencies are never imported at module import.
"""

from __future__ import annotations

import logging

__version__ = "0.1.4"
LOG = logging.getLogger("ComfyUI-KittenTTS2-Pro")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
REGISTRATION_ERRORS = []


def _load_nodes():
    from .kitten_pro.nodes import NODES

    for key, title, cls in NODES:
        NODE_CLASS_MAPPINGS[key] = cls
        NODE_DISPLAY_NAME_MAPPINGS[key] = title


try:
    _load_nodes()
except Exception as exc:
    LOG.exception("KittenTTS2-Pro node registration failed")
    REGISTRATION_ERRORS.append(f"{type(exc).__name__}: {exc}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
