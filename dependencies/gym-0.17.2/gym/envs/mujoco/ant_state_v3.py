import numpy as np
from gym import utils
from gym.envs.mujoco import state_mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class StateAntEnv_v3(state_mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_state.xml',
                 ctrl_cost_weight=0.3,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        state_mujoco_env.MujocoEnv.__init__(self, xml_file, 10)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, desired_joint_state):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        qp_before = self.sim.data.qpos.flat[7:]
        self.do_simulation_state(desired_joint_state, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        qp_after = self.sim.data.qpos.flat[7:]

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # ctrl_cost = self.control_cost(action) #TODO:
        # ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.get_actuator_force()))
        ctrl_cost = self._ctrl_cost_weight * np.square(qp_after-qp_before).sum()

        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def low_controller(self, full_desired_state):
        desired_joint_state = full_desired_state[7:15]
        return desired_joint_state

    def state_step(self, full_desired_state):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        qp_before = self.sim.data.qpos.flat[7:]
        desired_joint_state = self.low_controller(full_desired_state)
        self.do_simulation_state(desired_joint_state, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        qp_after = self.sim.data.qpos.flat[7:]

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # ctrl_cost = self.control_cost(action) #TODO:
        # ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.get_actuator_force()))
        ctrl_cost = self._ctrl_cost_weight * np.square(qp_after-qp_before).sum()

        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info


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
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
