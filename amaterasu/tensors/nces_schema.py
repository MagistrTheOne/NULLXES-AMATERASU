"""NCES 128-d node feature layout. Unused dims are zero; validity is a separate mask."""

POS = slice(0, 3)
ROT6D = slice(3, 9)
LIN_VEL = slice(9, 12)
ANG_VEL = slice(12, 15)
NODE_VALID_IN_FEAT = slice(15, 16)
GRASP = slice(16, 18)
CONTACT_BIN = slice(18, 19)
WRENCH = slice(19, 22)
GRAVITY = slice(22, 25)
SUPPORT = slice(25, 27)
MOMENTUM = slice(27, 30)
FRAME_FLAGS = slice(30, 32)
RESERVED = slice(32, 128)
D_NCES_IN = 128
