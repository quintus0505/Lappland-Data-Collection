import numpy as np
import we_envs
from gym import utils
from gym.envs.mujoco import state_mujoco_env

class StateAntEnv(state_mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        state_mujoco_env.MujocoEnv.__init__(self, 'ant_state.xml', 10)
        utils.EzPickle.__init__(self)


    def state_step(self, desired_state):
        xposbefore = self.get_body_com("torso")[0]
        qp_before = self.sim.data.qpos.flat[7:]
        reward_sim_error = self.do_simulation_state(desired_state, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        qp_after = self.sim.data.qpos.flat[7:]

        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(qp_after-qp_before).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward + reward_sim_error
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def step(self, desired_state):
        xposbefore = self.get_body_com("torso")[0]
        qp_before = self.sim.data.qpos.flat[7:]
        reward_sim_error = self.do_simulation_state(desired_state, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        qp_after = self.sim.data.qpos.flat[7:]

        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(qp_after-qp_before).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward + reward_sim_error
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def get_full_state(self):
        torso_pos = self.sim.data.qpos.flat[:3], #[3]
        torso_orientation = self.sim.data.qpos.flat[3:7] #[4]
        joint_angle = self.sim.data.qpos.flat[7:] #[8]

        torso_vel = self.sim.data.qvel.flat[:3] #[3]
        torso_angular_vel = self.sim.data.qvel.flat[3:6] #[3]
        joint_vel = self.sim.data.qvel.flat[6:] #[8]
        return {'torso_pos':torso_pos, 'torso_orientation':torso_orientation, 'joint_angle':joint_angle,
                'torso_vel':torso_vel, 'torso_angular_vel':torso_angular_vel, 'joint_vel':joint_vel}

    def get_full_state_joint(self):
        position_actuator_1 = self.sim.model.joint_name2id('hip_1')
        return self.sim.data.qpos.flat[position_actuator_1:position_actuator_1+8]

    def get_actuator_force(self):
        return self.sim.data.qfrc_actuator.flat # actuator forces?

    def get_cfrc_ext(self):
        return self.sim.data.cfrc_ext.flat # com-based external force on body

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat, #TODO:
            self.sim.data.cfrc_ext.flat
        ])
    # https://github.com/openai/gym/issues/585

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
