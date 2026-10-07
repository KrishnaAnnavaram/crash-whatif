"""Settings from environment variables and the default counterfactual space."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RESAMPLE_CHOICES = ("none", "class_weight", "oversample", "smotenc")
TARGET_CHOICES = ("binary", "three")

# Columns that a counterfactual can change. Driver, place, weather and time of the
# week stay fixed: a "what if" over those columns gives no action that a person can take.
DEFAULT_ACTIONABLE = (
    "abs",
    "esc",
    "tcs",
    "tpms",
    "airbags",
    "safety_rating",
    "vehicle_type",
    "transmission",
    "time_of_day",
)


class ConfigError(ValueError):
    """An environment variable has a value that the code cannot use."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Read ``KEY=VALUE`` lines from a local ``.env`` file. Existing variables win."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _get(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    data_path: str = ""  # empty: synthetic data
    out_dir: Path = field(default_factory=lambda: Path("reports"))
    resample: str = "class_weight"
    target: str = "binary"
    n_queries: int = 50
    actionable: tuple[str, ...] = DEFAULT_ACTIONABLE

    def __post_init__(self) -> None:
        if self.resample not in RESAMPLE_CHOICES:
            raise ConfigError(f"CRASH_WHATIF_RESAMPLE must be one of {RESAMPLE_CHOICES}, got {self.resample!r}")
        if self.target not in TARGET_CHOICES:
            raise ConfigError(f"CRASH_WHATIF_TARGET must be one of {TARGET_CHOICES}, got {self.target!r}")
        if self.n_queries < 1:
            raise ConfigError("CRASH_WHATIF_N_QUERIES must be at least 1")
        if not self.actionable:
            raise ConfigError("CRASH_WHATIF_ACTIONABLE must name at least one column")

    @classmethod
    def from_env(cls) -> "Settings":
        actionable = tuple(c.strip() for c in _get("CRASH_WHATIF_ACTIONABLE", ",".join(DEFAULT_ACTIONABLE)).split(",") if c.strip())
        return cls(
            seed=_int("CRASH_WHATIF_SEED", 42),
            data_path=_get("CRASH_WHATIF_DATA", ""),
            out_dir=Path(_get("CRASH_WHATIF_OUT", "reports")),
            resample=_get("CRASH_WHATIF_RESAMPLE", "class_weight"),
            target=_get("CRASH_WHATIF_TARGET", "binary"),
            n_queries=_int("CRASH_WHATIF_N_QUERIES", 50),
            actionable=actionable,
        )
