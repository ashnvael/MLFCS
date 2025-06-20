from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

@dataclass
class Config:
    WEALTH: float = field(default=1e7, metadata={"docs": "Initial wealth for the liquidity provider."})
    # TAU: float = field(default=0.003, metadata={"docs": "Proportional fee (swap fee) of the pool."})
    PATH_FAKE_DATA: Path = field(
        default=Path(__file__).resolve().parents[1] / "data" / "fake_data",
        metadata={"docs": "Relative path to the fake data folder"}
    )
    N_DAYS: int = field(default=100, metadata={"docs": "Number of days for fake data generation."})

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}:"
        for key, value in self.__dict__.items():
            string += f"\n* {key} = {value}"
        return string

    def __repr__(self) -> str:
        return self.__str__()
