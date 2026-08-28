from amaterasu.data.adapters.agency import record_to_sample as agency_record
from amaterasu.data.adapters.bones_seed import record_to_sample as bones_record
from amaterasu.data.adapters.egocentric import record_to_sample as ego_record
from amaterasu.data.adapters.hifi_umi import record_to_sample as hifi_record
from amaterasu.data.adapters.robot_traj import record_to_sample as robot_record
from amaterasu.data.adapters.simulation import record_to_sample as sim_record

ADAPTERS = {
    "bones-seed": bones_record,
    "hifi-umi-2k": hifi_record,
    "droid": robot_record,
    "oxe": robot_record,
    "ego4d": ego_record,
    "simulation": sim_record,
    "agency": agency_record,
}


def dispatch(source: str, record: dict):
    key = source.strip().lower()
    if key not in ADAPTERS:
        raise KeyError(f"no adapter for {source}")
    return ADAPTERS[key](record)
