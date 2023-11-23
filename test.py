from enum import Enum, unique
from pathlib import Path
from random import choices
from typing import Final

import click
from joblib import Parallel, delayed
from rich.progress import track

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    visualize_scenario,
)
from av2.map.map_api import ArgoverseStaticMap

from pathlib import Path

def generate_scenario_visualization(scenario_path: Path) -> None:
        """Generate and save dynamic visualization for a single Argoverse scenario.

        NOTE: This function assumes that the static map is stored in the same directory as the scenario file.

        Args:
            scenario_path: Path to the parquet file corresponding to the Argoverse scenario to visualize.
        """
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = (
            scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        )
        viz_save_path = Path(f"{scenario_id}.mp4")

        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        visualize_scenario(scenario, static_map, viz_save_path)


test = Path("../argoverse2/val/80032fab-8e7f-47ca-aac4-921d5a5c2edf/scenario_80032fab-8e7f-47ca-aac4-921d5a5c2edf.parquet")

generate_scenario_visualization(test)