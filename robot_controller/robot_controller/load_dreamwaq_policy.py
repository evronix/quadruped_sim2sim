import os
import sys
import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
import sys, types



class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4, device="cpu"):
        super().__init__()
        self.device = device
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False).to(self.device)
        self.var = nn.Parameter(torch.ones(shape), requires_grad=False).to(self.device)
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(axis=0).to(self.device)
        batch_std = x.std(axis=0).to(self.device)
        batch_count = x.shape[0]
        batch_var = batch_std ** 2
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count
        self.mean.data = new_mean
        self.var.data = new_var
        self.count = tot_count

dummy_rsl_rl = types.ModuleType("rsl_rl")
dummy_utils = types.ModuleType("rsl_rl.utils")
dummy_rms = types.ModuleType("rsl_rl.utils.rms")
dummy_rms.RunningMeanStd = RunningMeanStd 
dummy_utils.rms = dummy_rms
dummy_rsl_rl.utils = dummy_utils
sys.modules["rsl_rl"] = dummy_rsl_rl
sys.modules["rsl_rl.utils"] = dummy_utils
sys.modules["rsl_rl.utils.rms"] = dummy_rms

def quat_rotate_inverse(q, v):
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


class CENet(nn.Module):
    def __init__(self, input_dim=225, latent_dim=19):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 3 + 32)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu

    def inference(self, obs_history_225):
        h = self.encoder(obs_history_225)
        est_vel = h[:, :3]
        half = (h.shape[1] - 3) // 2
        mu = h[:, 3:3 + half]
        logvar = h[:, 3 + half:]
        z = self.reparameterize(mu, logvar)
        latent_19 = torch.cat([est_vel, z], dim=-1)
        return latent_19


class Actor(nn.Module):
    def __init__(self, input_dim=64, output_dim=12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim)
        )

    def act_inference(self, obs_1d):
        return self.mlp(obs_1d)


def load_learned_models(pt_file, cenet, actor, device='cpu'):
    checkpoint = torch.load(pt_file, map_location=device)
    if 'cenet_state_dict' in checkpoint:
        full_sd = checkpoint['cenet_state_dict']
        filtered_enc = {k: v for k, v in full_sd.items() if k.startswith('encoder.')}
        cenet.load_state_dict(filtered_enc, strict=False)
    else:
        print("[Warn] no cenet_state_dict in checkpoint")
    model_sd = checkpoint.get('model_state_dict', {})
    actor_sd = {k.replace('actor.', 'mlp.'): v for k, v in model_sd.items() if k.startswith('actor.')}
    actor.load_state_dict(actor_sd, strict=False)
    rms_info = checkpoint.get('rms', None)
    return rms_info


class AgentNode(Node):
    def __init__(self, pt_file):
        super().__init__('agent_node')
        self.device = 'cpu'
        self.cenet = CENet(225, 19).to(self.device)
        self.actor = Actor(64, 12).to(self.device)
        self.rms_info = load_learned_models(pt_file, self.cenet, self.actor, device=self.device)
        self.cenet.eval()
        self.actor.eval()
        self.obs_rms = None
        if self.rms_info is not None and 'obs_rms' in self.rms_info:
            shape_dim = 45
            self.obs_rms = RunningMeanStd((shape_dim,), device=self.device)
            self.obs_rms.mean.data = self.rms_info['obs_rms'].mean.to(self.device)
            self.obs_rms.var.data = self.rms_info['obs_rms'].var.to(self.device)
            self.obs_rms.count = float(self.rms_info['obs_rms'].count)
            self.get_logger().info(f"[obs_rms] loaded shape=({shape_dim}), count={self.obs_rms.count}")
        self.num_actions = 12
        self.default_dof_pos = torch.tensor(
            [[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]],
            dtype=torch.float
        )
        self.dof_pos = torch.zeros((1, 12), dtype=torch.float)
        self.q_des = torch.zeros((1, 12), dtype=torch.float)
        self.dof_vel = torch.zeros((1, 12), dtype=torch.float)
        self.actions = torch.zeros((1, 12), dtype=torch.float)
        self.torques = torch.zeros((1, 12), dtype=torch.float)
        self.action_clip = 100.0
        self.counter = 0
        
        self.action_scale = 0.25
        self.Kp = 28
        self.Kd = 0.7
        self.obs_history_length = 5
        self.num_obs_45 = 45
        self.obs_history_buf = torch.zeros((1, self.obs_history_length, self.num_obs_45), dtype=torch.float)
        self.base_ang_vel = torch.zeros((1, 3), dtype=torch.float)
        self.base_quat = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        self.gravity_vector = torch.tensor([[0, 0, -1]], dtype=torch.float)
        self.projected_gravity = torch.tensor([[0, 0, 0]], dtype=torch.float)
        self.commands = torch.tensor([[0.15, 0.0, 0.0]], dtype=torch.float)
        self.commands_sub = self.create_subscription(
            PoseStamped, '/robot_velocity_command', self.velocity_command_callback, 10
        )
        self.joint_effort_pub = self.create_publisher(
            Float64MultiArray, '/joint_effort_controller/commands', 10
        )
        self.pos_pub = self.create_publisher(Float64MultiArray, '/policy', 10)

        self.imu_sub = self.create_subscription(
            Imu, '/imu_plugin/out', self.imu_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.timer = self.create_timer(0.005, self.control_loop) # policy dt : 0.02, PD controller dt : 0.005
        self.get_logger().info(f"AgentNode init done. checkpoint={pt_file}")

    def velocity_command_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.commands = torch.tensor([[x, y, z]], dtype=torch.float)

    def imu_callback(self, msg: Imu):
        self.base_quat = torch.tensor(
            [[msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]],
            dtype=torch.float
        )
        self.projected_gravity = quat_rotate(self.base_quat, self.gravity_vector)
        self.base_ang_vel[0, 0] = msg.angular_velocity.x
        self.base_ang_vel[0, 1] = msg.angular_velocity.y
        self.base_ang_vel[0, 2] = msg.angular_velocity.z

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) >= 12:
            self.dof_pos = torch.tensor([[
                msg.position[9], msg.position[6], msg.position[7],
                msg.position[0], msg.position[2], msg.position[8],
                msg.position[5], msg.position[3], msg.position[10],
                msg.position[1], msg.position[4], msg.position[11]
            ]], dtype=torch.float)
            self.dof_vel = torch.tensor([[
                msg.velocity[9], msg.velocity[6], msg.velocity[7],
                msg.velocity[0], msg.velocity[2], msg.velocity[8],
                msg.velocity[5], msg.velocity[3], msg.velocity[10],
                msg.velocity[1], msg.velocity[4], msg.velocity[11]
            ]], dtype=torch.float)

    def build_obs_45(self):
        part1 = self.base_ang_vel
        part2 = self.projected_gravity
        part3 = self.commands
        part4 = self.dof_pos - self.default_dof_pos
        part5 = self.dof_vel
        part6 = self.actions
        return torch.cat([part1, part2, part3, part4, part5, part6], dim=-1)

    def update_ring_buffer(self, raw_obs_45):
        temp = self.obs_history_buf[:, 1:, :].clone()
        self.obs_history_buf[:, :-1, :].copy_(temp)
        self.obs_history_buf[:, -1, :] = raw_obs_45

    def control_loop(self): # step
        if self.counter % 4 == 0:
            
            obs_45_raw = self.build_obs_45()
            if self.obs_rms is not None:
                obs_45_norm = (obs_45_raw - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)
            else:
                obs_45_norm = obs_45_raw.clone()
            self.update_ring_buffer(obs_45_raw)
            if self.obs_rms is not None:
                obs_history_3d_norm = (self.obs_history_buf - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-6)
            else:
                obs_history_3d_norm = self.obs_history_buf.clone()
            obs_history_225_norm = obs_history_3d_norm.reshape(1, -1)
            with torch.no_grad():
                h = self.cenet.encoder(obs_history_225_norm)
                est_vel = h[:, :3]
                half = (h.shape[1] - 3) // 2
                mu = h[:, 3:3 + half]
                latent_19 = torch.cat([est_vel, mu], dim=-1)
                actor_obs = torch.cat([obs_45_norm, latent_19], dim=-1)
            with torch.no_grad():
                raw_actions = self.actor.act_inference(actor_obs)
            raw_actions = torch.clip(raw_actions, -self.action_clip, self.action_clip)
            actions_scaled = raw_actions * self.action_scale
            actions_scaled[:, [0, 3, 6, 9]] *= 0.5
            self.q_des =  self.default_dof_pos + actions_scaled
            #print(f"des", self.default_dof_pos + actions_scaled)
            
            self.pos_pub.publish(Float64MultiArray(data=[float(x) for x in self.q_des[0]])) 
            
            self.actions = raw_actions.clone()
            self.counter = 0
        self.torques = self.Kp * (self.q_des - self.dof_pos) - self.Kd * self.dof_vel
        self.publish_torques(self.torques[0])
        self.counter +=1

    def publish_torques(self, torques_1d):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in torques_1d]
        self.joint_effort_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    pt_file = "/home/kjh/quadruped_sim2sim/policy_model/dreamwaq/model_5000.pt"
    if not os.path.isfile(pt_file):
        print(f"{pt_file} not found!")
        sys.exit(0)
    node = AgentNode(pt_file)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
