import numpy as np
from gym import utils
from gym.envs.mujoco import state_mujoco_env_swimmer_pid

class StateSwimmerEnvPid(state_mujoco_env_swimmer_pid.MujocoEnv, utils.EzPickle):
    def __init__(self):
        state_mujoco_env_swimmer_pid.MujocoEnv.__init__(self, 'swimmer_state_pid.xml', 10)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def low_controller(self, full_desired_state):
        desired_joint_state = full_desired_state[3:5] #TODO:
        return np.array(desired_joint_state)

    def state_step(self, full_desired_state):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        qp_before = self.sim.data.qpos.flat[3:5]

        desired_joint_state = self.low_controller(full_desired_state)
        self.do_simulation_state(desired_joint_state, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        qp_after = self.sim.data.qpos.flat[3:5]

        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(qp_after-qp_before).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def get_full_state(self):
        # torso_pos = self.sim.data.qpos.flat[:3], #[3]
        # torso_orientation = self.sim.data.qpos.flat[3:7] #[4]
        # rot2 and rot3 are the actuation joint
        rot1_joint_id = self.sim.model.joint_name2id('rot')
        joint_angle = self.sim.data.qpos.flat[rot1_joint_id:rot1_joint_id+3]

        torso_slide_joint_id = self.sim.model.joint_name2id('slider1')
        slide_angle = self.sim.data.qpos.flat[torso_slide_joint_id:torso_slide_joint_id+2]

        joint_vel = self.sim.data.qvel.flat[rot1_joint_id:rot1_joint_id+3]
        slide_vel = self.sim.data.qvel.flat[torso_slide_joint_id:torso_slide_joint_id+2]

        return {'joint_angle':joint_angle, 'slide_angle':slide_angle,
                'joint_vel':joint_vel, 'slide_vel':slide_vel}

    def desire_state_range(self):
        position_actuator_1 = self.sim.model.actuator_name2id('pos_rot2')
        ctrlrange = self.sim.model.actuator_ctrlrange
        position_ctrlrange = ctrlrange[position_actuator_1:position_actuator_1+2]
        desire_state_range = np.ones_like(self._get_obs())
        desire_state_range[3:5]= position_ctrlrange[:, 1]
        return desire_state_range

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
