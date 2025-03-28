import torch
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped

import time
import pickle as pkl
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Header


torch.set_printoptions(threshold=2500)

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit', map_location=torch.device('cpu'))
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit', map_location=torch.device('cpu'))

    def policy(obs, info):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        self.logdir = "/home/kjh/sim2sim_quadruped/src/policy_model/walk-these-ways/025417.456545"
        with open(self.logdir + "/parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            self.cfg = pkl_cfg["Cfg"]

        self.policy = load_policy(self.logdir)
        self.policy_info = {}

        self.action_scale = 0.25
        self.hip_scale_reduction = 0.5
        self.torques = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
        self.default_dof_pos = torch.tensor([[0.000, 0.8000, -1.5000, -0.000, 0.8000, -1.5000, 0.000, 1.0000,
                                              -1.5000, -0.000, 1.0000, -1.5000]])
        self.default_deg_pos = torch.rad2deg(self.default_dof_pos)
        self.dof_pos = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.dof_vel = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.obs_scales_dof_vel = 0.05
        self.obs_scales_dof_vel = 0.05
        self.obs_scales_dof_pos = 1.0

        self.base_quat = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        self.gravity_vector = torch.tensor([[0, 0, -1]], dtype=torch.float)
        self.projected_gravity = torch.tensor([[0, 0, 0]], dtype=torch.float)

        self.commands = torch.tensor([[0.0, 0.0, 0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 0.5, 0.04, 0.0, 0.0, 0.25, 0.3720, 0.0098]], dtype=torch.float)
        self.commands_scale = torch.tensor([2.0000, 2.0000, 0.2500, 2.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                            0.1500, 0.3000, 0.3000, 1.0000, 1.0000, 1.0000])

        self.dt = 0.02
        self.num_envs = 1
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)
        self.gait_indices = torch.tensor([0.])

        self.noise_scale_vec = torch.tensor([0.0500, 0.0500, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                            0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100,
                                            0.0100, 0.0100, 0.0100, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750,
                                            0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0000, 0.0000, 0.0000,
                                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
        self.obs_history_length = 30
        self.num_obs = 70
        self.num_envs = 1
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, dtype=torch.float, requires_grad=False)
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(7)]

        self.actions = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.last_actions = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        # /joint_position_controller/commands
        self.joint_effort_publisher = self.create_publisher(
            Float64MultiArray, '/joint_effort_controller/commands', 10)
        
        self.velocity_subscriber = self.create_subscription(
            PoseStamped, '/robot_velocity_command', self.velocity_command_callback, 10)        
        self.imu_subscriber = self.create_subscription(
            Imu, '/imu_plugin/out', self.imu_callback, 10)
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.timer = self.create_timer(0.02, self.control_loop)
        self.last_time = time.time()


    def velocity_command_callback(self, msg: PoseStamped):
        new_values = torch.tensor([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=torch.float)
        self.commands[0, :3] = new_values


    def imu_callback(self, data):
        self.base_quat = torch.tensor([[data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w]], dtype=torch.float)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vector)

    def joint_state_callback(self, data:JointState):
            self.dof_pos = torch.tensor([[data.position[9],data.position[6],data.position[7],
                                         data.position[0],data.position[2],data.position[8],
                                         data.position[5],data.position[3],data.position[10],
                                         data.position[1],data.position[4],data.position[11]]])
            self.dof_vel = torch.tensor([[data.velocity[9],data.velocity[6],data.velocity[7],
                                         data.velocity[0],data.velocity[2],data.velocity[8],
                                         data.velocity[5],data.velocity[3],data.velocity[10],
                                         data.velocity[1],data.velocity[4],data.velocity[11]]])
    def compute_clock_inputs(self):
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        durations = self.commands[:, 8]

        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        return self.clock_inputs

    def compute_observation(self):
        self.obs_buf = torch.cat((self.projected_gravity,                                           
                                  self.commands * self.commands_scale,                              
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales_dof_pos,    
                                  self.dof_vel * self.obs_scales_dof_vel,                           
                                  self.actions), dim=-1)                                            

        self.obs_buf = torch.cat((self.obs_buf, self.last_actions), dim=-1) 

        self.compute_clock_inputs()
        self.obs_buf = torch.cat((self.obs_buf, self.clock_inputs), dim=-1)
        privileged_obs = torch.tensor([[1. , -1.]], dtype=torch.float)
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], self.obs_buf), dim=-1)
        obs = {'obs': self.obs_buf, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
        self.last_actions = self.actions
        return obs

    def step(self, actions, hard_reset=False):
        clip_actions = 10
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        actions_scaled = self.actions[:, :12] * self.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.hip_scale_reduction
        self.joint_pos_target = actions_scaled + self.default_dof_pos
        self.torques = 20 * 1 * (self.joint_pos_target- self.dof_pos + 0) - 0.5 * 1 * self.dof_vel

        self.publish_joint_trajectory(self.torques)

    def publish_joint_trajectory(self, joint_positions):

        msg = Float64MultiArray()
        msg.data = [float(pos) for pos in self.torques[0]]
        self.joint_effort_publisher.publish(msg)

    def control_loop(self):
        current_time = time.time()
        self.last_time = current_time

        obs = self.compute_observation()
        actions = self.policy(obs, self.policy_info)
        self.step(actions)

def main(args=None):
    rclpy.init(args=args)
    agent_node = AgentNode()
    rclpy.spin(agent_node)
    agent_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()