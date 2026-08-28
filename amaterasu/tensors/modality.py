from enum import IntEnum


class ModalityId(IntEnum):
    VISION = 0
    LANGUAGE = 1
    PHYSICAL = 2
    AGENCY = 3


class IntentKind(IntEnum):
    ACT_0 = 0
    ACT_1 = 1
    ACT_2 = 2
    ACT_3 = 3
    ACT_4 = 4
    ACT_5 = 5
    ACT_6 = 6
    ACT_7 = 7
    OBSERVE = 8
    HOLD = 9
    WAIT = 10


class GateDecision(IntEnum):
    ALLOW = 0
    DEFER = 1
    BLOCK = 2


class NodeType(IntEnum):
    ROOT = 0
    HEAD = 1
    TORSO = 2
    L_ARM = 3
    R_ARM = 4
    L_HAND = 5
    R_HAND = 6
    L_LEG = 7
    R_LEG = 8
    EXTRA = 9
