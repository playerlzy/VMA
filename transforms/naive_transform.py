import torch

from utils.geometry import wrap_angle

class NaiveTransform:

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int,
                 a2m_radius: int,
                 a2a_radius: int,
                 m2m_radius: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.a2m_radius = a2m_radius
        self.a2a_radius = a2a_radius
        self.m2m_radius = m2m_radius

    def __call__(self, data):
        
        agent_origin = data['global_position'][:, self.num_historical_steps - 1]
        agent_head = data["global_heading"][:, self.num_historical_steps - 1]
        map_origin = data["lane_ctr"]
        map_theta = data["lane_theta"]

        ##normalize to itself 
        #agent feature
        agent_cos, agent_sin = agent_head.cos(), agent_head.sin()
        agent_rot_mat = agent_head.new_zeros(40, 2, 2)
        agent_rot_mat[:, 0, 0] = agent_cos
        agent_rot_mat[:, 0, 1] = -agent_sin
        agent_rot_mat[:, 1, 0] = agent_sin
        agent_rot_mat[:, 1, 1] = agent_cos
        data["agent_feature"] = agent_origin.new_zeros(40, self.num_historical_steps, 6)
        data["agent_feature"][:, :, :2] = torch.matmul(
            data["global_position"][:, :self.num_historical_steps] - 
            agent_origin.unsqueeze(1), agent_rot_mat
        )
        rel_head = wrap_angle(data["global_heading"][:, :self.num_historical_steps] -
                              agent_head.unsqueeze(-1))
        data["agent_feature"][:, :, 2] = rel_head.cos()
        data["agent_feature"][:, :, 3] = rel_head.sin()
        data["agent_feature"][:, :, 4:6] = torch.matmul(
            data["global_vel"][:, :self.num_historical_steps], agent_rot_mat
        )

        data["target"] = agent_origin.new_zeros(40, self.num_future_steps, 3)
        data["target"][:, :, :2] = torch.matmul(
            data["global_position"][:, self.num_historical_steps:] - 
            agent_origin.unsqueeze(1), agent_rot_mat
        )
        data["target"][:, :, 2] = wrap_angle(data["global_heading"][:, self.num_historical_steps:] -
                                             agent_head.unsqueeze(-1))
        #map feature
        map_cos, map_sin = map_theta.cos(), map_theta.sin()
        map_rot_mat = map_theta.new_zeros(128, 2, 2)
        map_rot_mat[:, 0, 0] = map_cos
        map_rot_mat[:, 0, 1] = -map_sin
        map_rot_mat[:, 1, 0] = map_sin
        map_rot_mat[:, 1, 1] = map_cos
        data["map_feature"] = map_origin.new_zeros(128, 19, 6)
        data["map_feature"][:, :, :2] = torch.matmul(
            data["lane_position"][:, :19] - 
            map_origin.unsqueeze(1), map_rot_mat
        )
        data["map_feature"][:, :, 2:4] = torch.matmul(
            data["lane_position"][:, 1:] - 
            map_origin.unsqueeze(1), map_rot_mat
        )
        global_theta = torch.atan2(
            data["lane_position"][:, 1:, 1] - data["lane_position"][:, :19, 1],
            data["lane_position"][:, 1:, 0] - data["lane_position"][:, :19, 0],
        )
        rel_theta = wrap_angle(global_theta - map_theta.unsqueeze(-1))
        data["map_feature"][:, :, 4] = rel_theta.cos()
        data["map_feature"][:, :, 5] = rel_theta.sin()
        
        #m2m distance
        lane_map = data["lane_map"] + 1
        data["m2m_feature"] = map_origin.new_zeros(128, 128, 4)
        m2m_rel_pos = torch.matmul(
            map_origin.unsqueeze(1) - map_origin.unsqueeze(0), map_rot_mat
        )
        m2m_rel_theta = wrap_angle(
            map_theta.unsqueeze(1) - map_theta.unsqueeze(0)
        )
        data["m2m_feature"][:, :, :2] = m2m_rel_pos
        data["m2m_feature"][:, :, 2] = m2m_rel_theta.cos()
        data["m2m_feature"][:, :, 3] = m2m_rel_theta.sin()
        data["m2m_edge"] = torch.max(lane_map, 
                                     torch.norm(m2m_rel_pos, dim=-1) < self.m2m_radius
        )
        data["m2m_edge"].fill_diagonal_(0)

        #a2m distance
        data["a2m_feature"] = agent_origin.new_zeros(40, 128, 4)
        a2m_rel_pos = torch.matmul(
            agent_origin.unsqueeze(1) - map_origin.unsqueeze(0), agent_rot_mat
        )
        a2m_rel_theta = wrap_angle(
            agent_head.unsqueeze(1) - map_theta.unsqueeze(0)
        )
        data["a2m_feature"][:, :, :2] = a2m_rel_pos
        data["a2m_feature"][:, :, 2] = a2m_rel_theta.cos()
        data["a2m_feature"][:, :, 3] = a2m_rel_theta.sin()
        data["a2m_edge"] = (torch.norm(a2m_rel_pos, dim=-1) < self.a2m_radius)

        #a2a distance
        data["a2a_feature"] = agent_origin.new_zeros(40, 40, 4)
        a2a_rel_pos = torch.matmul(
            agent_origin.unsqueeze(1) - agent_origin.unsqueeze(0), agent_rot_mat
        )
        a2a_rel_theta = wrap_angle(
            agent_head.unsqueeze(1) - agent_head.unsqueeze(0)
        )
        data["a2a_feature"][:, :, :2] = a2a_rel_pos
        data["a2a_feature"][:, :, 2] = a2a_rel_theta.cos()
        data["a2a_feature"][:, :, 3] = a2a_rel_theta.sin()
        data["a2a_edge"] = (torch.norm(a2a_rel_pos, dim=-1) < self.a2a_radius)

        return data