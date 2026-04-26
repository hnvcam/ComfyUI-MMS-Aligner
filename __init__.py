"""ComfyUI-MMS-Aligner — force-align audio with text into SRT using Meta's MMS."""

from .nodes.mms_aligner import MMSAligner

NODE_CLASS_MAPPINGS = {
    "MMSAligner": MMSAligner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMSAligner": "MMS Audio-Text Aligner (SRT)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
