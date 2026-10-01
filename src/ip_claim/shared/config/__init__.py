"""Configuration boundary for the live ip_claim stack.

Train and eval knobs live in ``ip_claim.ssv.config`` and
``ip_claim.collision.config``. Hub tokens resolve via OmegaConf
``${oc.env:HF_TOKEN}`` at YAML load into Pydantic ``SecretStr`` fields.
"""

from __future__ import annotations

__all__: list[str] = []
