from __future__ import annotations

from amaterasu.tensors.modality import NodeType

NODE_ORDER = (
    NodeType.ROOT,
    NodeType.HEAD,
    NodeType.TORSO,
    NodeType.L_ARM,
    NodeType.R_ARM,
    NodeType.L_HAND,
    NodeType.R_HAND,
    NodeType.L_LEG,
    NodeType.R_LEG,
)
