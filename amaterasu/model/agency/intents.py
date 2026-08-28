from __future__ import annotations

from amaterasu.tensors.modality import IntentKind

ACT_KINDS = (
    IntentKind.ACT_0,
    IntentKind.ACT_1,
    IntentKind.ACT_2,
    IntentKind.ACT_3,
    IntentKind.ACT_4,
    IntentKind.ACT_5,
    IntentKind.ACT_6,
    IntentKind.ACT_7,
)
NOOP_KINDS = (IntentKind.OBSERVE, IntentKind.HOLD, IntentKind.WAIT)
ALL_KINDS = ACT_KINDS + NOOP_KINDS
