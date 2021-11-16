import numpy as np
from gym import utils
from gym.envs.mujoco import state_mujoco_env_swimmer

class StateSwimmerPositionEnv(state_mujoco_env_swimmer.MujocoEnv, utils.EzPickle):
    def __init__(self):
        state_mujoco_env_swimmer.MujocoEnv.__init__(self, 'swimmer_state_absposition.xml', 10)
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
        desired_joint_state = full_desired_state[4:6] #TODO:
        return desired_joint_state

    def state_step(self, full_desired_state):
        ctrl_cost_coeff = 0.0001
        # xposbefore = self.sim.data.qpos[0]
        qp_before = self.sim.data.qpos.flat[3:5]

        desired_joint_state = self.low_controller(full_desired_state)
        self.do_simulation_state(desired_joint_state, self.frame_skip)
        # xposafter = self.sim.data.qpos[0]
        qp_after = self.sim.data.qpos.flat[3:5]

        # reward_fwd = (xposafter - xposbefore) / self.dt
        reward_fwd = 0
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

        head_site_id = self.sim.model.site_name2id("root1")
        head2_site_id = self.sim.model.site_name2id("root2")
        head_pos = self.sim.data.site_xpos[head_site_id].flat
        head2_pos = self.sim.data.site_xpos[head2_site_id].flat

        return {'joint_angle':joint_angle, 'slide_angle':slide_angle,
                'joint_vel':joint_vel, 'slide_vel':slide_vel,
                'head2_pos': head2_pos, 'head2_pos':head2_pos
                }

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        joint_angle = qpos.flat[3:5]
        qvel = qvel.flat[3:5]

        head_site_id = self.sim.model.site_name2id("root1")
        head2_site_id = self.sim.model.site_name2id("root2")
        head_pos = self.sim.data.site_xpos[head_site_id].flat[0:2]
        head2_pos = self.sim.data.site_xpos[head2_site_id].flat[0:2]

        return np.concatenate([head_pos, head2_pos, joint_angle, qvel])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
