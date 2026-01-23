from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TABLES = ("D_Design", "D_Pop", "D_Drug", "R_Study", "R_Arm_Study")
ALL_TABLES = (
    "D_Design",
    "D_Pop",
    "D_Drug",
    "R_Study",
    "R_Study_Endpoint",
    "R_Arm_Study",
    "R_Arm_Study_Endpoint",
)


@dataclass(frozen=True)
class PipelineConfig:
    group_dir: Path
    output_dir: Path
    tables: tuple[str, ...] = DEFAULT_TABLES

    @classmethod
    def from_group(cls, group_dir: str | Path, output_dir: str | Path | None = None) -> "PipelineConfig":
        group_path = Path(group_dir)
        if output_dir is None:
            output_dir = group_path / "_ml_pipeline"
        return cls(group_dir=group_path, output_dir=Path(output_dir))
