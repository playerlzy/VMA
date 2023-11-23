from pathlib import Path

import pandas as pd
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.map.map_api import ArgoverseStaticMap

OBJECT_TYPE_MAP = {
    ObjectType.VEHICLE.value: 0,
    ObjectType.PEDESTRIAN.value: 1,
    ObjectType.MOTORCYCLIST.value: 2,
    ObjectType.CYCLIST.value: 3,
    ObjectType.BUS.value: 4,
    ObjectType.STATIC.value: 5,
    ObjectType.BACKGROUND.value: 6,
    ObjectType.CONSTRUCTION.value: 7,
    ObjectType.RIDERLESS_BICYCLE.value: 8,
    ObjectType.UNKNOWN.value: 9,
}


OBJECT_TYPE_MAP_COMBINED = {
    ObjectType.VEHICLE.value: 0,
    ObjectType.PEDESTRIAN.value: 1,
    ObjectType.MOTORCYCLIST.value: 2,
    ObjectType.CYCLIST.value: 2,
    ObjectType.BUS.value: 0,
    ObjectType.STATIC.value: 3,
    ObjectType.BACKGROUND.value: 3,
    ObjectType.CONSTRUCTION.value: 3,
    ObjectType.RIDERLESS_BICYCLE.value: 3,
    ObjectType.UNKNOWN.value: 3,
}

LaneTypeMap = {
    LaneType.VEHICLE.value: 0,
    LaneType.BIKE.value: 1,
    LaneType.BUS.value: 2,
    'crosswalk': 3,
}

LaneMarkTypeMap = {
    LaneMarkType.DASH_SOLID_YELLOW.value: 1,
    LaneMarkType.DASH_SOLID_WHITE.value: 2,
    LaneMarkType.DASHED_WHITE.value: 3,
    LaneMarkType.DASHED_YELLOW.value: 4,
    LaneMarkType.DOUBLE_SOLID_YELLOW.value: 5,
    LaneMarkType.DOUBLE_SOLID_WHITE.value: 6,
    LaneMarkType.DOUBLE_DASH_YELLOW.value: 7,
    LaneMarkType.DOUBLE_DASH_WHITE.value: 8,
    LaneMarkType.SOLID_YELLOW.value: 9,
    LaneMarkType.SOLID_WHITE.value: 10,
    LaneMarkType.SOLID_DASH_WHITE.value: 11,
    LaneMarkType.SOLID_DASH_YELLOW.value: 12,
    LaneMarkType.SOLID_BLUE.value: 13,
    LaneMarkType.NONE.value: 14,
    LaneMarkType.UNKNOWN.value: 15,
}

IsIntersectionMap = {
    True: 0,
    False: 1,
    None: 2,
}

AdjMap = {
    'pred': 1,
    'succ': 2,
    'left': 3,
    'right': 4,
}

def load_av2_df(scenario_file: Path):
    scenario_id = scenario_file.stem.split("_")[-1]
    df = pd.read_parquet(scenario_file)
    static_map = ArgoverseStaticMap.from_json(
        scenario_file.parents[0] / f"log_map_archive_{scenario_id}.json"
    )

    return df, static_map, scenario_id
