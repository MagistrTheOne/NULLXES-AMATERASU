from __future__ import annotations

COMMERCIAL_BLOCKED = frozenset(
    {
        "agibot",
        "agibot-world",
        "egodex",
        "ego4d",
        "epic",
        "bones-seed",
    }
)

RESEARCH_ALLOWED_IF_TERMS = frozenset(
    {
        "hifi-umi-2k",
        "droid",
        "oxe",
        "bones-seed",
        "agibot",
        "egodex",
        "ego4d",
        "robomind",
        "bridge",
        "agency",
        "simulation",
    }
)


def license_ok(source: str, commercial: bool) -> bool:
    key = source.strip().lower()
    if commercial:
        if key in COMMERCIAL_BLOCKED:
            return False
        if key == "bones-seed":
            return False
        return True
    return key in RESEARCH_ALLOWED_IF_TERMS or key.startswith("nullxes") or key in {"agency", "simulation", "hifi-umi-2k", "droid"}


def filter_sources(sources: list[str], commercial: bool) -> list[bool]:
    return [license_ok(s, commercial) for s in sources]
