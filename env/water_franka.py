import numpy as np
import genesis as gs
import torch

class WaterFrankaEnv:
    def __init__(self, vis, device, num_envs=1):
        self.device = device
        self.action_space = 8  
        self.state_dim = 3  
        assert num_envs == 1, "Liquid only supports num_envs=1"
        self.num_envs = num_envs
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=4e-3,
                substeps=10,
            ),
            sph_options=gs.options.SPHOptions(
                lower_bound=(0.3, -0.4, 0.0),
                upper_bound=(1.3, 0.4, 15.0),
                particle_size=0.01,
            ),
            vis_options=gs.options.VisOptions(
                visualize_sph_boundary=True,
                show_world_frame = True,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
            show_viewer=vis,
            renderer=gs.renderers.RayTracer(),
            viewer_options = gs.options.ViewerOptions(
                res           = (1280, 960),
                camera_pos    = (3.5, 0.0, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 40,
                max_FPS       = 60,
            ),
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.franka = self.scene.add_entity(
            # gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda.xml"),
            gs.morphs.MJCF(file="assets/xml/franka_emika_panda/panda_with_spoon.xml"),
            # gs.morphs.URDF(file="../assets/urdf/panda_bullet/panda.urdf", fixed=True), 
        )
        self.liquid = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(),
            morph=gs.morphs.Box(
                pos=(0.8, 0.0, 0.15),
                size=(1.0, 0.8, 0.3),
            ),
            surface=gs.surfaces.Water(
                # color=(0.0, 0.0, 1.0),
                # opacity=0.5,
                vis_mode="recon",  # or "recon"
            ),
        )
        self.cam = self.scene.add_camera(
            res    = (640, 480),
            pos    = (0.0, 3.0, 2.5),
            lookat = (0, 0, 0.5),
            fov    = 30,
            GUI    = False,
            spp    = 512,
        )
        self.recording = False
        # 添加漂浮球
        # self.floating_ball = self.scene.add_entity(
        #     material=gs.materials.Rigid(rho=500),  # 固体材料
        #     morph=gs.morphs.Sphere(
        #         pos=(0.6, 0.1, 0.1),  # 设置球的初始位置
        #         radius=0.03,           # 设置球的半径
        #     ),
        #     surface=gs.surfaces.Default(
        #         color=(1.0, 0.0, 0.0),  # 红色球
        #         vis_mode="visual",    # 可视化为粒子
        #     ),
        # )
        
        # 添加沉底球
        self.sinking_ball = self.scene.add_entity(
            material=gs.materials.Rigid(rho=1500),  # 固体材料
            morph=gs.morphs.Sphere(
                pos=(0.65, -0.1, 0.02), # 设置球的初始位置
                radius=0.01,            # 设置球的半径
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 1.0, 0.0),  # 绿色球
                vis_mode="visual",    # 可视化为粒子
            ),
        )
        """
        print(dir(gs.materials.SPH.Liquid()))
        ['__class__', '__colorized__repr__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_data_oriented', '_exponent', '_gamma', '_is_debugger', '_mu', '_repr_brief', '_repr_briefer', '_repr_type', '_rho', '_sampler', '_stiffness', '_uid', 'exponent', 'gamma', 'mu', 'rho', 'sampler', 'stiffness', 'uid']
        print(dir(self.liquid))
        ['__class__', '__colorized__repr__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_add_to_solver', '_add_to_solver_', '_add_vverts_to_solver', '_assert_active', '_ckpt', '_data_oriented', '_idx', '_init_particles_offset', '_is_debugger', '_kernel_get_mass', '_kernel_get_particles', '_material', '_morph', '_n_particles', '_need_skinning', '_particle_size', '_particle_start', '_particles', '_queried_states', '_repr_brief', '_repr_briefer', '_repr_type', '_scene', '_sim', '_solver', '_surface', '_tgt', '_tgt_buffer', '_tgt_keys', '_uid', '_vface_start', '_vfaces', '_vmesh', '_vvert_start', '_vverts', 'activate', 'active', 'add_grad_from_state', 'clear_grad', 'collect_output_grads', 'deactivate', 'get_frame', 'get_mass', 'get_particles', 'get_state', 'idx', 'init_ckpt', 'init_particles', 'init_sampler', 'init_tgt_keys', 'init_tgt_vars', 'is_built', 'load_ckpt', 'material', 'morph', 'n_particles', 'n_vfaces', 'n_vverts', 'particle_end', 'particle_size', 'particle_start', 'process_input', 'process_input_grad', 'reset_grad', 'sample', 'sampler', 'save_ckpt', 'scene', 'set_active', 'set_pos', 'set_pos_grad', 'set_position', 'set_vel', 'set_vel_grad', 'set_velocity', 'sim', 'solver', 'surface', 'uid', 'vface_end', 'vface_start', 'vmesh', 'vvert_end', 'vvert_start']
        print(self.liquid.n_particles)
        print(self.liquid.get_state().pos) # same as: print(self.liquid.get_particles())
        """
        self.scene.build()
        self.build_env()
    
    def build_env(self):
        self.motors_dof = torch.arange(7).to(self.device)
        # self.fingers_dof = torch.arange(7, 9).to(self.device)

        self.end_effector = self.franka.get_link("spoon")

        self.pos = torch.tensor([0.65, 0.0, 0.135], dtype=torch.float32, device=self.device)
        self.quat = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=self.device)
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos = self.pos,
            quat = self.quat,
        )

        # # 检查 qpos 的形状
        # print("qpos shape:", self.qpos.shape)
        # print("qpos:", self.qpos)

        # # 检查机械臂的自由度数
        # print("Number of DOFs:", self.franka.n_dofs)

        franka_lower_limits, franka_upper_limits = self.franka.get_dofs_limit()
        self.qpos = 0.7 * (franka_upper_limits + franka_lower_limits)

        self.franka.set_qpos(self.qpos)
        self.scene.step()
        self.franka.control_dofs_position(self.qpos, self.motors_dof)

    def reset(self):
        self.build_env()
        gripper_position = self.franka.get_link("spoon").get_pos()  # 使用 spoon 的位置
        states = gripper_position.clone().detach().unsqueeze(0).to(self.device)  # 修复 UserWarning
        return states

    def start_recording(self):
        self.cam.start_recording()
        self.recording = True

    def stop_recording(self):
        self.cam.stop_recording(save_to_filename='video.mp4', fps=60)
        self.recording = False

    def step(self, actions, i):
        if actions is None:
            self.scene.step()
            return None
        # finger_pos = torch.tensor([0.04, 0.04], device=self.device)
        spoon_pos = self.franka.get_link("spoon").get_pos()  # 勺子的位置
        ball_pos = self.sinking_ball.get_pos()       # 沉底球的位置

        distance_0 = torch.norm(spoon_pos - ball_pos)
        height_0 = ball_pos[2]

        pos = self.pos.clone()
        if actions == 1: # Close gripper
            pass
        elif actions == 2: # Lift gripper 
            # finger_pos[:] = 0
            pos[2] = 1.0
        elif actions == 3: # Lower gripper
            pos[2] = 0
        elif actions == 4: # Move left
            pos[0] -= 0.05
        elif actions == 5: # Move right
            pos[0] += 0.05
        elif actions == 6: # Move forward
            pos[1] -= 0.05
        elif actions == 7: # Move backward
            pos[1] += 0.05

        self.pos = pos
        self.qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=pos,
            quat=self.quat,
        )

        self.franka.control_dofs_position(self.qpos, self.motors_dof)
        # self.franka.control_dofs_position(finger_pos, self.fingers_dof)
        self.scene.step()

        gripper_position = (self.franka.get_link("spoon").get_pos() + self.franka.get_link("right_finger").get_pos()) / 2
        states = gripper_position.clone().detach().unsqueeze(0).to(self.device)

        # rewards = self.liquid.get_particles()[:, 2].max() 
        # 获取勺子和沉底球的位置
        spoon_pos = self.franka.get_link("spoon").get_pos()  # 勺子的位置
        ball_pos = self.sinking_ball.get_pos()       # 沉底球的位置

        # 计算勺子和球之间的距离
        height = ball_pos[2]
        distance = torch.norm(spoon_pos - ball_pos)  # 欧几里得距离

        # 定义奖励：距离越小奖励越高
        # threshold_distance = 0.01
        # if distance >= threshold_distance:
        #     rewards = torch.exp(-distance)  # 距离越小奖励越高
        # else:
        #     rewards = ball_pos[2]
        # rewards = 100.0 * (distance_0 - distance)  + 1000.0 * (height - height_0)
        # rewards = 100.0 * (distance_0 - distance)
        threshold_dist = 0.01
        if distance > threshold_dist:
            rewards = 100 * (distance_0 - distance)
        else:
            rewards = 100 * (distance_0 - distance) + 1000 * (height - height_0)
        dones = False
        rewards = torch.tensor([rewards], device=self.device)
        dones = torch.tensor([dones], device=self.device)
        if self.recording:
            # self.cam.set_pose(
            #     pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
            #     lookat = (0, 0, 0.5),
            # )
            self.cam.render()
        
        return states, rewards, dones

if __name__ == "__main__":
    gs.init()
    env = WaterFrankaEnv(vis=True, device=gs.device, num_envs=1)
    horizon = 10000
    up = False
    for i in range(horizon):
        if i % 200 == 0:
            up = not up
        if i < 100:
            actions = None
        else:
            if up:
                actions = torch.ones(1, device="cuda:0") * 2
            else:
                actions = torch.ones(1, device="cuda:0") * 3
        env.step(actions)