from __future__ import annotations

SPECIAL_TOKEN_NAMES = (
    "<|am_pad|>",
    "<|am_bos|>",
    "<|am_eos|>",
    "<|am_null_instruction|>",
    "<|am_act|>",
    "<|am_observe|>",
    "<|am_hold|>",
    "<|am_wait|>",
    "<|am_allow|>",
    "<|am_defer|>",
    "<|am_block|>",
    "<|am_mem_write|>",
    "<|am_mem_read|>",
    "<|am_reserved_13|>",
    "<|am_reserved_14|>",
    "<|am_reserved_15|>",
)

SPECIAL_TO_ID = {name: i for i, name in enumerate(SPECIAL_TOKEN_NAMES)}
NULL_INSTRUCTION_ID = SPECIAL_TO_ID["<|am_null_instruction|>"]
PAD_ID = SPECIAL_TO_ID["<|am_pad|>"]
BOS_ID = SPECIAL_TO_ID["<|am_bos|>"]
EOS_ID = SPECIAL_TO_ID["<|am_eos|>"]
