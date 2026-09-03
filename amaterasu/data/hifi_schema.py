"""HiFi-UMI-2K LeRobot-v3 frame table. Circuit-0 reads parquet only — never MP4."""

HIFI_REPO = "simple-world-lab/HiFi-UMI-2K"
HIFI_LICENSE = "CC-BY-4.0"

# observation.state / action: right 10-d then left 10-d
# each hand: xyz(3) + rot6d(6) + gripper_rad(1)
RIGHT = slice(0, 10)
LEFT = slice(10, 20)
XYZ = slice(0, 3)
ROT6D = slice(3, 9)
GRIP = 9

PARQUET_COLUMNS = (
    "observation.state",
    "observation.state_valid",
    "action",
    "action_valid",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "valid.frame",
)
