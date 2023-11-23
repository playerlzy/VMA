import traceback
from pathlib import Path
from typing import List

import av2.geometry.interpolate as interp_utils
import numpy as np
import torch
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneMarkType

from utils.av2_data_utils import (
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    LaneMarkTypeMap,
    IsIntersectionMap,
    AdjMap,
    load_av2_df,
)

def get_list_idx(lane_ids: List, valid_lane_ids: List, segment_id):
    if segment_id not in lane_ids:
        return None
    idx = lane_ids.index(segment_id)
    if idx not in valid_lane_ids:
        return None
    return valid_lane_ids.index(idx)

class Av2Extractor:
    def __init__(
        self,
        save_path: Path = None,
        mode: str = "train",
        max_num_agent: int = 40,
        max_num_lane: int = 128,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.max_num_agent = max_num_agent
        self.max_num_lane = max_num_lane

    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file, self.max_num_agent, self.max_num_lane)

    def process(self, raw_path: str, max_num_agent: int, max_num_lane: int, agent_id=None):
        df, am, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]
        agent_id = df["focal_track_id"].values[0]

        local_df = df[df["track_id"] == agent_id].iloc
        origin_focal = torch.tensor(
            [local_df[49]["position_x"], local_df[49]["position_y"]], dtype=torch.float
        )

        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]

        #filter agents by sorting the distance
        dis_to_origin = []
        actor_ids = []
        for actor_id, actor_df in cur_df.groupby("track_id"):
            actor_ids.append(actor_id)
            cur_pos = torch.tensor(
                [actor_df["position_x"].values[0], actor_df["position_y"].values[0]]
            )
            dis_to_origin.append(np.linalg.norm(cur_pos - origin_focal))

        dis_to_origin = np.array(dis_to_origin)
        ids_by_sort = np.argsort(dis_to_origin)
        actor_ids = [actor_ids[ids_by_sort[i]] 
                     for i in range(min(len(ids_by_sort), max_num_agent))
        ]

        df = df[df["track_id"].isin(actor_ids)]

        global_position = torch.zeros(max_num_agent, 110, 2, dtype=torch.float)
        object_type = torch.zeros(max_num_agent, dtype=torch.uint8)
        object_category = torch.zeros(max_num_agent, dtype=torch.uint8)
        global_heading = torch.zeros(max_num_agent, 110, dtype=torch.float)
        global_velocity = torch.zeros(max_num_agent, 110, 2, dtype=torch.float)
        padding_mask = torch.ones(max_num_agent, 110, dtype=torch.bool)
        
        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]
            object_type[node_idx] = OBJECT_TYPE_MAP_COMBINED[
                actor_df["object_type"].values[0]
            ]
            object_category[node_idx] = actor_df["object_category"].values[0]

            padding_mask[node_idx, node_steps] = False
            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["position_x"].values, actor_df["position_y"].values],
                    axis=-1,
                )
            ).float()
            heading = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values
            ).float()

            global_position[node_idx, node_steps, :2] = pos_xy
            global_heading[node_idx, node_steps] = heading
            global_velocity[node_idx, node_steps, :2] = velocity

        (
            lane_position,
            lane_ctr,
            lane_theta,
            lane_type,
            lane_is_intersection,
            lane_width,
            lane_mask,
            lane_left_type,
            lane_right_type,
            lane_map
        ) = self.get_map_features(am, origin_focal, max_num_lane)

        return {
            "global_position": global_position,
            "object_type": object_type,
            "object_category": object_category,
            "global_heading": global_heading,
            "global_vel": global_velocity,
            "padding_mask": padding_mask,
            "lane_position": lane_position,
            "lane_ctr": lane_ctr,
            "lane_theta": lane_theta,
            "lane_type": lane_type,
            "lane_width": lane_width,
            "is_intersection": lane_is_intersection,
            "lane_mask": lane_mask,
            "lane_left_type": lane_left_type,
            "lane_right_type": lane_right_type,
            "lane_map": lane_map,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }

    @staticmethod
    def get_map_features(map_api: ArgoverseStaticMap,
                         origin: int,
                         max_num_lanes: int):
        lane_segment_ids = map_api.get_scenario_lane_segment_ids()
        cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())
        lane_ids = lane_segment_ids + cross_walk_ids
        num_lanes = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        lane_position = torch.zeros(max_num_lanes, 20, 2, dtype=torch.float)
        lane_ctr = torch.zeros(max_num_lanes, 2, dtype=torch.float)
        lane_theta = torch.zeros(max_num_lanes, dtype=torch.float)
        lane_type = torch.zeros(max_num_lanes, dtype=torch.uint8)
        lane_is_intersection = torch.zeros(max_num_lanes, dtype=torch.uint8)
        lane_width = torch.zeros(max_num_lanes, dtype=torch.float)
        lane_mask = torch.ones(max_num_lanes, dtype=torch.bool)
        lane_left_type = torch.zeros(max_num_lanes, dtype=torch.uint8)
        lane_right_type = torch.zeros(max_num_lanes, dtype=torch.uint8)
        lane_map = torch.zeros(max_num_lanes, max_num_lanes, dtype=torch.uint8)

        #filter lanes by sorting the distance
        lane_dis_to_origin = torch.zeros(num_lanes, dtype=torch.float)

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = lane_ids.index(lane_segment.id)
            lane_centerline, _ = interp_utils.compute_midpoint_line(
                left_ln_boundary=lane_segment.left_lane_boundary.xyz,
                right_ln_boundary=lane_segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            min_dist = torch.norm(
                lane_centerline - origin, dim=-1).min(dim=0).values
            lane_dis_to_origin[lane_segment_idx] = min_dist

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = lane_ids.index(crosswalk.id)
            edge1 = crosswalk.edge1.xyz
            edge2 = crosswalk.edge2.xyz
            lane_centerline, _ = interp_utils.compute_midpoint_line(
                left_ln_boundary=edge1,
                right_ln_boundary=edge2,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            min_dist = torch.norm(
                lane_centerline - origin, dim=-1).min(dim=0).values
            lane_dis_to_origin[crosswalk_idx] = min_dist
            lane_dis_to_origin[crosswalk_idx + len(cross_walk_ids)] = min_dist

        ids_by_sort = np.argsort(lane_dis_to_origin)
        valid_lane_ids = [ids_by_sort[i] for i in range(min(max_num_lanes, num_lanes))]

        #get polyline features
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = lane_ids.index(lane_segment.id)
            if lane_segment_idx not in valid_lane_ids:
                continue
            lane_segment_idx = valid_lane_ids.index(lane_segment_idx)
            lane_centerline, width = interp_utils.compute_midpoint_line(
                left_ln_boundary=lane_segment.left_lane_boundary.xyz,
                right_ln_boundary=lane_segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            lane_position[lane_segment_idx] = lane_centerline
            lane_ctr[lane_segment_idx] = (lane_centerline[9] + lane_centerline[10]) / 2
            lane_theta[lane_segment_idx] = torch.atan2(
                lane_centerline[10, 1] - lane_centerline[9, 1],
                lane_centerline[10, 0] - lane_centerline[9, 0],
            )
            lane_width[lane_segment_idx] = width
            lane_mask[lane_segment_idx] = False
            lane_is_intersection[lane_segment_idx] = \
                IsIntersectionMap[lane_segment.is_intersection]
            lane_type[lane_segment_idx] = LaneTypeMap[lane_segment.lane_type]
            lane_left_type[lane_segment_idx] = LaneMarkTypeMap[lane_segment.left_mark_type]
            lane_right_type[lane_segment_idx] = LaneMarkTypeMap[lane_segment.right_mark_type]

        for crosswalk in map_api.get_scenario_ped_crossings():
            cross_walk_idx = lane_ids.index(crosswalk.id)
            lane_centerline, width = interp_utils.compute_midpoint_line(
                left_ln_boundary=crosswalk.edge1.xyz,
                right_ln_boundary=crosswalk.edge2.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            if cross_walk_idx in valid_lane_ids:
                lane_segment_idx = valid_lane_ids.index(cross_walk_idx)
                lane_position[lane_segment_idx] = lane_centerline
                lane_ctr[lane_segment_idx] = (lane_centerline[9] + lane_centerline[10]) / 2
                lane_theta[lane_segment_idx] = torch.atan2(
                    lane_centerline[10, 1] - lane_centerline[9, 1],
                    lane_centerline[10, 0] - lane_centerline[9, 0],
                )
                lane_width[lane_segment_idx] = width
                lane_mask[lane_segment_idx] = False
                lane_is_intersection[lane_segment_idx] = IsIntersectionMap[None]
                lane_type[lane_segment_idx] = LaneTypeMap["crosswalk"]
                lane_left_type[lane_segment_idx] = LaneMarkTypeMap[LaneMarkType.UNKNOWN.value]
                lane_right_type[lane_segment_idx] = LaneMarkTypeMap[LaneMarkType.UNKNOWN.value]

            if cross_walk_idx + len(cross_walk_ids) in valid_lane_ids:
                lane_segment_idx = valid_lane_ids.index(cross_walk_idx + len(cross_walk_ids))
                lane_centerline = lane_centerline.flip(dims=[0])
                lane_position[lane_segment_idx] = lane_centerline
                lane_ctr[lane_segment_idx] = (lane_centerline[9] + lane_centerline[10]) / 2
                lane_theta[lane_segment_idx] = torch.atan2(
                    lane_centerline[10, 1] - lane_centerline[9, 1],
                    lane_centerline[10, 0] - lane_centerline[9, 0],
                )
                lane_width[lane_segment_idx] = width
                lane_mask[lane_segment_idx] = False
                lane_is_intersection[lane_segment_idx] = IsIntersectionMap[None]
                lane_type[lane_segment_idx] = LaneTypeMap["crosswalk"]
                lane_left_type[lane_segment_idx] = LaneMarkTypeMap[LaneMarkType.UNKNOWN.value]
                lane_right_type[lane_segment_idx] = LaneMarkTypeMap[LaneMarkType.UNKNOWN.value]

        #process nbr relations
        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = lane_ids.index(lane_segment.id)
            if lane_segment_idx not in valid_lane_ids:
                continue
            lane_segment_idx = valid_lane_ids.index(lane_segment_idx)
            #pred
            for pred in lane_segment.predecessors:
                pred_idx = get_list_idx(lane_ids, valid_lane_ids, pred)
                if pred_idx is not None:
                    lane_map[lane_segment_idx, pred_idx] = AdjMap['pred']
            #succ
            for succ in lane_segment.successors:
                succ_idx = get_list_idx(lane_ids, valid_lane_ids, succ)
                if succ_idx is not None:
                    lane_map[lane_segment_idx, succ_idx] = AdjMap['succ']
            #left
            if lane_segment.left_neighbor_id is not None:
                left_idx = get_list_idx(lane_ids, valid_lane_ids, lane_segment.left_neighbor_id)
                if left_idx is not None:
                    lane_map[lane_segment_idx, left_idx] = AdjMap['left']
            #right
            if lane_segment.right_neighbor_id is not None:
                right_idx = get_list_idx(lane_ids, valid_lane_ids, lane_segment.right_neighbor_id)
                if right_idx is not None:
                    lane_map[lane_segment_idx, right_idx] = AdjMap['right']


        return (
            lane_position,
            lane_ctr,
            lane_theta,
            lane_type,
            lane_is_intersection,
            lane_width,
            lane_mask,
            lane_left_type,
            lane_right_type,
            lane_map
        )