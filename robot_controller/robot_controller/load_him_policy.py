import os
import sys
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# =============================================================================
# 1. Utils & Math Helper
# =============================================================================

def quat_rotate_inverse(q, v):
    """
    Rotate vector v by inverse of quaternion q.
    q: [w, x, y, z] (IsaacGym style usually [x, y, z, w], but we handle conversion in callback)
    We assume q is [x, y, z, w] here to match standard ROS/scipy conventions if consistent.
    However, IsaacGym 'quat_rotate_inverse' expects [x, y, z, w].
    Let's ensure consistent usage.
    """
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def quat_rotate(q, v):
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        super().__init__()
        self.device = device
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False).to(self.device)
        self.var = nn.Parameter(torch.ones(shape), requires_grad=False).to(self.device)
        self.count = epsilon

# =============================================================================
# 2. Network Definitions (Updated for HIM Policy)
# =============================================================================

class HIMEstimator(nn.Module):
    """
    Structure based on model_5000.pt inspection:
    input(270) -> 128 -> 64 -> 19
    Output 19 splits into: Vel(3) + Latent(16)
    """
    def __init__(self, input_dim=270, output_dim=19, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs_history):
        # obs_history: (Batch, 270)
        out = self.encoder(obs_history)
        vel_est = out[:, :3]
        latent = out[:, 3:]
        # Normalize latent vector (Crucial step in HIM)
        latent = F.normalize(latent, dim=-1, p=2)
        return vel_est, latent

class Actor(nn.Module):
    """
    Structure based on model_5000.pt inspection:
    input(64) -> 512 -> 256 -> 128 -> 12
    Input 64 comes from: Obs(45) + Vel(3) + Latent(16)
    """
    def __init__(self, input_dim=64, output_dim=12, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs_input):
        return self.mlp(obs_input)

# =============================================================================
# 3. Model Loader
# =============================================================================

def load_checkpoint(pt_file, estimator, actor, device='cpu'):
    print(f"Loading model from {pt_file}...")
    checkpoint = torch.load(pt_file, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # 1. Load Estimator
    # Map 'estimator.encoder.X...' keys to our estimator.encoder
    est_dict = {}
    for k, v in state_dict.items():
        if k.startswith('estimator.encoder.'):
            new_key = k.replace('estimator.', '') # Remove 'estimator.' prefix
            est_dict[new_key] = v
    estimator.load_state_dict(est_dict)
    print("Estimator loaded successfully.")

    # 2. Load Actor
    # Map 'actor.X...' keys to our actor.mlp (since we used nn.Sequential named mlp)
    act_dict = {}
    for k, v in state_dict.items():
        if k.startswith('actor.'):
            # model_state_dict keys are like 'actor.0.weight'
            # Our class uses 'mlp.0.weight'
            new_key = k.replace('actor.', 'mlp.')
            act_dict[new_key] = v
    actor.load_state_dict(act_dict)
    print("Actor loaded successfully.")

    # 3. Load Normalization Stats (if available)
    obs_rms = None
    
    return obs_rms

# =============================================================================
# 4. ROS2 Agent Node
# =============================================================================

class AgentNode(Node):
    def __init__(self, pt_file):
        super().__init__('agent_node')
        self.device = 'cpu'
        
        # -- Hyperparameters --
        self.num_obs = 45
        self.history_len = 6 
        self.num_actions = 12
        self.action_scale = 0.25 # From Go1RoughCfg.control
        
        # [FIX] Separate scales for Lin Vel and Ang Vel
        self.obs_scales = {
            'lin_vel': 2.0,
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
        }
        self.clip_obs = 100.0
        self.clip_actions = 100.0
        
        # -- Robot State --
        # Order: FL, FR, RL, RR (Isaac Gym Standard)
        self.default_dof_pos = torch.tensor(
            [[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]],
            dtype=torch.float, device=self.device
        )
        
        self.dof_pos = torch.zeros((1, 12), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((1, 12), dtype=torch.float, device=self.device)
        self.base_ang_vel = torch.zeros((1, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.tensor([[0, 0, 0, 1]], dtype=torch.float, device=self.device) 
        self.gravity_vec = torch.tensor([[0, 0, -1]], dtype=torch.float, device=self.device)
        self.commands = torch.zeros((1, 3), dtype=torch.float, device=self.device) 
        self.actions = torch.zeros((1, 12), dtype=torch.float, device=self.device) 
        self.q_des = self.default_dof_pos.clone()
        
        # -- Buffers --
        self.obs_history_buf = torch.zeros((1, self.history_len, self.num_obs), device=self.device)
        self.history_initialized = False

        # -- Models --
        self.estimator = HIMEstimator().to(self.device)
        self.actor = Actor().to(self.device)
        self.obs_rms = load_checkpoint(pt_file, self.estimator, self.actor, self.device)
        self.estimator.eval()
        self.actor.eval()

        # -- PD Gains (From Paper) --
        self.Kp = 40.0 
        self.Kd = 1.0

        # -- Joint Mapping --
        self.isaac_joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        self.joint_indices = None 

        # -- ROS Interface --
        self.create_subscription(PoseStamped, '/robot_velocity_command', self.cmd_callback, 10)
        self.create_subscription(Imu, '/imu_plugin/out', self.imu_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_effort_controller/commands', 10)
        
        # -- Timers --
        self.dt = 0.02 # Policy DT
        self.decimation = 10 # Sim Decimation
        self.sim_dt = 0.002 # Sim DT (500Hz)
        self.counter = 0
        self.create_timer(self.sim_dt, self.control_loop)
        
        self.get_logger().info("HIM Agent Node Initialized with Corrected Scales.")

    def cmd_callback(self, msg):
        # x, y, yaw
        self.commands[0, 0] = msg.pose.position.x
        self.commands[0, 1] = msg.pose.position.y
        self.commands[0, 2] = msg.pose.position.z 

    def imu_callback(self, msg):
        self.base_quat[0, 0] = msg.orientation.x
        self.base_quat[0, 1] = msg.orientation.y
        self.base_quat[0, 2] = msg.orientation.z
        self.base_quat[0, 3] = msg.orientation.w
        
        self.base_ang_vel[0, 0] = msg.angular_velocity.x
        self.base_ang_vel[0, 1] = msg.angular_velocity.y
        self.base_ang_vel[0, 2] = msg.angular_velocity.z

    def joint_callback(self, msg):
        # Dynamic Mapping based on Joint Names
        if self.joint_indices is None:
            try:
                self.joint_indices = [msg.name.index(n) for n in self.isaac_joint_names]
                self.get_logger().info(f"Joint indices mapped: {self.joint_indices}")
            except ValueError as e:
                self.get_logger().error(f"Joint mapping failed: {e}")
                return

        if len(msg.position) >= 12:
            pos = torch.tensor([msg.position[i] for i in self.joint_indices], device=self.device)
            vel = torch.tensor([msg.velocity[i] for i in self.joint_indices], device=self.device)
            self.dof_pos[0] = pos
            self.dof_vel[0] = vel

    def get_observations(self):
        projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # [FIX] Create specific scale tensor for commands
        # lin_vel(2.0), lin_vel(2.0), ang_vel(0.25)
        cmd_scale = torch.tensor(
            [self.obs_scales['lin_vel'], self.obs_scales['lin_vel'], self.obs_scales['ang_vel']],
            device=self.device
        )
        
        obs_list = [
            self.commands * cmd_scale,                          # 3 (Fix applied here)
            self.base_ang_vel * self.obs_scales['ang_vel'],     # 3
            projected_gravity,                                  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'], # 12
            self.dof_vel * self.obs_scales['dof_vel'],          # 12
            self.actions                                        # 12
        ]
        obs = torch.cat(obs_list, dim=-1)
        
        if self.clip_obs is not None:
            obs = torch.clip(obs, -self.clip_obs, self.clip_obs)
            
        return obs

    def control_loop(self):
            # 1. Policy Inference Step (Runs at lower frequency)
            if self.counter % self.decimation == 0:
                with torch.no_grad():
                    current_obs = self.get_observations() # Shape (1, 45)
                    
                    # [중요] Normalization
                    if self.obs_rms is not None:
                        norm_obs = (current_obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)
                    else:
                        norm_obs = current_obs 

                    # [FIXED] History Buffer Logic (Isaac Gym과 동일하게 수정)
                    # 기존: shifts=-1 (왼쪽으로 밀기) -> 수정: shifts=1 (오른쪽으로 밀기)
                    # 기존: index -1 (맨 뒤에 넣기) -> 수정: index 0 (맨 앞에 넣기)
                    # 결과: [Current, t-1, t-2, ... ] 순서 유지
                    
                    if not self.history_initialized:
                        # 처음 시작할 때는 전체를 현재 값으로 채움 (Zero padding 방지)
                        for i in range(self.history_len):
                            self.obs_history_buf[:, i, :] = norm_obs
                        self.history_initialized = True
                    else:
                        # 오른쪽으로 한 칸 밈 (가장 오래된 데이터 삭제됨)
                        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=1, dims=1)
                        # 맨 앞에 최신 데이터 삽입
                        self.obs_history_buf[:, 0, :] = norm_obs
                    
                    # Flatten history for Estimator (1, 270)
                    # view를 하면 [batch, 0번데이터, 1번데이터...] 순서로 펴지므로
                    # [Current, t-1, t-2 ...] 순서가 되어 Estimator 입력과 일치해짐
                    history_flat = self.obs_history_buf.view(1, -1)

                    # D. Estimator Inference
                    est_vel, latent = self.estimator(history_flat)
                    
                    # E. Actor Inference
                    actor_input = torch.cat([norm_obs, est_vel, latent], dim=-1)
                    
                    raw_actions = self.actor(actor_input)
                    
                    self.actions = raw_actions # Store for next obs
                    
                    # Action Scaling (No Hip Reduction as per Config)
                    scaled_actions = raw_actions * self.action_scale
                    
                    self.q_des = self.default_dof_pos + scaled_actions

            # 2. PD Control Step (Runs at high frequency 500Hz)
            torques = self.Kp * (self.q_des - self.dof_pos) - self.Kd * self.dof_vel
            torques = torch.clip(torques, -80.0, 80.0) 
            
            msg = Float64MultiArray()
            msg.data = torques[0].tolist()
            self.joint_pub.publish(msg)
            
            self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    pt_file = "/home/kjh/quadruped_sim2sim/policy_model/him/model_10000.pt" 
    
    if not os.path.isfile(pt_file):
        print(f"Error: Model file {pt_file} not found!")
        sys.exit(1)
        
    node = AgentNode(pt_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()