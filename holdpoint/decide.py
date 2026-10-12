"""Width band + user threshold table -> one of four suggestions.

There is no built-in standard. The limit comes from a table the user supplies (YAML); the example table in
config/ is labelled "example values, not a standard".

    band entirely <= limit      -> accept_suggested
    band entirely  > limit      -> repair_suggested
    band straddles the limit    -> measure_with_gauge
    no band (no scale card)     -> width_not_measurable
A resolution-limited band (crack only a few pixels wide) never yields accept_suggested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ACCEPT = "accept_suggested"
REPAIR = "repair_suggested"
GAUGE = "measure_with_gauge"
NOT_MEASURABLE = "width_not_measurable"
STATES = (ACCEPT, REPAIR, GAUGE, NOT_MEASURABLE)


@dataclass
class ThresholdTable:
    name: str
    label: str
    limits_mm: dict[str, float]

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdTable":
        classes = d.get("classes") or {}
        if not classes:
            raise ValueError("threshold table has no classes")
        limits = {}
        for k, v in classes.items():
            lim = float(v["max_width_mm"])
            if lim <= 0:
                raise ValueError(f"class {k!r}: max_width_mm must be > 0")
            limits[k] = lim
        return cls(name=str(d.get("name", "unnamed table")), label=str(d.get("label", "")), limits_mm=limits)

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdTable":
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def limit(self, element_class: str) -> float:
        try:
            return self.limits_mm[element_class]
        except KeyError:
            raise KeyError(f"class {element_class!r} not in table {self.name!r}") from None


@dataclass
class Suggestion:
    state: str
    reason: str
    element_class: str
    limit_mm: float | None
    band_mm: tuple[float, float] | None
    table: str
    table_label: str

    def to_dict(self) -> dict:
        return asdict(self)


def suggest(band_mm: tuple[float, float] | None, element_class: str, table: ThresholdTable,
            resolution_limited: bool = False) -> Suggestion:
    def out(state, reason, limit=None):
        return Suggestion(state, reason, element_class, limit, tuple(band_mm) if band_mm else None,
                          table.name, table.label)

    if band_mm is None:
        return out(NOT_MEASURABLE, "no scale card in view, width not measured")
    limit = table.limit(element_class)
    lo, hi = band_mm
    if lo > hi:
        raise ValueError(f"band lower bound above upper bound: {band_mm}")
    if lo > limit:
        return out(REPAIR, "whole band above the limit", limit)
    if resolution_limited:
        return out(GAUGE, "crack only a few pixels wide; photo cannot resolve it", limit)
    if hi <= limit:
        return out(ACCEPT, "whole band at or below the limit", limit)
    return out(GAUGE, "band crosses the limit", limit)
