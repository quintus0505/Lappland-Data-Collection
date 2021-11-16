import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
import random

ADD_BONUS_REWARDS = True


class DoorEnvV3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0

        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.init_state_for_terminial_use['palm_pos'] = np.array([0, 0, 0])      
        self.demo_starting_state = []
        self.demo_terminal_state = []
        self.demo_num = 0
        self.random_index = 0
        self.demo_terminal_state.append(self.init_state_for_terminial_use)
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + '/../assets/mj_envs/DAPG_door.xml', 5)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[
        self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, 0:3] = np.array(
            [10, 0, 0])
        self.sim.model.actuator_gainprm[
        self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0') + 1, 0:3] = np.array(
            [1, 0, 0])
        self.sim.model.actuator_biasprm[
        self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, 0:3] = np.array(
            [0, -10, 0])
        self.sim.model.actuator_biasprm[
        self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0') + 1, 0:3] = np.array(
            [0, -1, 0])

        utils.EzPickle.__init__(self)
        # ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')

        self.ARTz_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARTz')]
        self.ARRx_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARRx')]
        self.ARRy_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARRy')]
        self.ARRz_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARRz')]

    def load_data(self, demo_starting_state, demo_terminal_state):
        self.demo_starting_state = demo_starting_state
        self.demo_terminal_state = demo_terminal_state
        self.demo_num = len(demo_starting_state)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        reward_tb = np.zeros(5, dtype=np.float)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]
        latch_pos = self.data.qpos[-1].ravel()
        door_hinge_pos = self.data.qpos[self.door_hinge_did].ravel().copy()

        '''old reward'''
        '''
        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False
        '''
        terminal_palm_pos = self.demo_terminal_state[self.random_index]['palm_pos']
        constrain = 0.05
        # get to handle
        reward = -0.5 * np.linalg.norm(palm_pos - handle_pos)
        reward_tb[0] = reward
        if door_pos<1.57-constrain:
            reward -= abs(1.57-constrain-door_pos)
            reward_tb[1] = -abs(1.57-constrain-door_pos)


        # velocity cost
        reward += -1e-5 * np.sum(self.data.qvel ** 2)
        reward_tb[2] = -1e-5 * np.sum(self.data.qvel ** 2)

        goal_achieved = True if  abs(1.57-door_pos) < constrain else False
        if goal_achieved:
            reward += 30.0
            reward_tb[3] = 30.0

        return ob, reward, False, dict(goal_achieved=goal_achieved, rewardtb=reward_tb)

    def step_statemode(self, desired_state):
        assert desired_state.shape == (28,)
        ctrlrange = self.sim.model.actuator_ctrlrange

        wrist_dof_idx0 = self.sim.model.actuator_name2id('A_ARTz')
        self.sim.data.ctrl[wrist_dof_idx0: wrist_dof_idx0 + 4] = desired_state[0:4]

        hand_dof_idx0 = self.sim.model.actuator_name2id('pos:A_WRJ1')
        self.sim.data.ctrl[hand_dof_idx0: hand_dof_idx0 + 24] = desired_state[4:]

        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        self.sim.step()

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        return np.concatenate(
            [qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos - handle_pos, [door_open]])

    def get_obs_utils(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]

        arm_Tz = self.data.qpos[self.ARTz_pos]
        arm_Rx = self.data.qpos[self.ARRx_pos]
        arm_Ry = self.data.qpos[self.ARRy_pos]
        arm_Rz = self.data.qpos[self.ARRz_pos]

        obs_dict = {'qp': qp[1:-2], 'arm': np.array([arm_Tz, arm_Rx, arm_Ry, arm_Rz]), 'latch_pos': latch_pos,
                    'door_pos': door_pos, 'palm_pos': palm_pos, 'handle_pos': handle_pos,
                    'palm_relative_pos': palm_pos - handle_pos,
                    'door_open': door_open}
        return obs_dict

    def reset_model(self):
        '''
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        '''
        self.random_index = random.randint(0, self.demo_num - 1)
        begin_state = self.demo_starting_state[self.random_index]
        terminal_state = self.demo_terminal_state[self.random_index]

        qp = begin_state['qpos']
        qv = begin_state['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = begin_state['door_body_pos']
        self.data.site_xpos[self.handle_sid] = begin_state['handle_pos']
        self.data.site_xpos[self.grasp_sid] = begin_state['palm_pos']

        latch_pos = self.data.qpos[-1].ravel()

        self.sim.forward()
        return self.get_obs()

    def get_target_rl_goal(self):
        terminal_state = self.demo_terminal_state[self.random_index]
        target_palm_pos = terminal_state['palm_pos']
        return target_palm_pos

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel().copy()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel().copy()
        door_hinge_pos = self.data.qpos[self.door_hinge_did].ravel().copy()
        latch_pos = self.data.qpos[-1]

        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos,
                    handle_pos=handle_pos, palm_pos=palm_pos, door_hinge_pos=door_hinge_pos, latch_pos=latch_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:  # door open for 25 steps
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage

    def render(self, mode='human'):
        self.mj_render()

    def reset(self):
        return self._reset()
