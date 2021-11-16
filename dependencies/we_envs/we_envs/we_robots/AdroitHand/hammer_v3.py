import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
from we_envs.we_robots.AdroitHand.utils.quatmath import *
import os
import random

ADD_BONUS_REWARDS = True


class HammerEnvV3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        self.init_state = dict()
        self.init_state['init_obj_pos'] = np.array([0, 0, 0])
        self.init_state['init_target_pos'] = np.array([0, 0, 0])
        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.init_state_for_terminial_use['palm_pos'] = np.array([0, 0, 0])      
        self.demo_starting_state = []
        self.demo_terminal_state = []
        self.demo_num = 0
        self.random_index = 0
        self.demo_terminal_state.append(self.init_state_for_terminial_use)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + '/../assets/mj_envs/DAPG_hammer.xml', 5)
        utils.EzPickle.__init__(self)

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

        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        reward_tb = np.zeros(5, dtype=np.float)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()
        '''       
        # get to hammer
        reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
        # take hammer head to nail
        reward -= np.linalg.norm((tool_pos - target_pos))
        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if (np.linalg.norm(target_pos - goal_pos) < 0.020):
                reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < 0.010):
                reward += 75
        '''

        terminal_palm_pos = self.demo_terminal_state[self.random_index]['palm_pos']        
        # get to hammer
        reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
        reward_tb[0] = reward
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        reward_tb[1] = -10 * np.linalg.norm(target_pos - goal_pos)

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.01 and \
                        np.linalg.norm(palm_pos - obj_pos) < 0.05 else False
        if goal_achieved:
            reward += 20.0
            reward_tb[2] = 20.0

        return ob, reward, False, dict(goal_achieved=goal_achieved, rewardtb=reward_tb)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_rot = quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = np.clip(self.sim.data.sensordata[self.sim.model.sensor_name2id('S_nail')], -1.0, 1.0)
        return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset_model(self):
        '''
        self.sim.reset()
        target_bid = self.model.body_name2id('nail_board')
        self.model.body_pos[target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        '''

        self.random_index = random.randint(0, self.demo_num - 1)
        begin_state = self.demo_starting_state[self.random_index]
        terminal_state = self.demo_terminal_state[self.random_index]

        qp = begin_state['qpos']
        qv = begin_state['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = begin_state['board_pos']
        self.data.site_xpos[self.target_obj_sid] = begin_state['target_pos']
        self.data.body_xpos[self.obj_bid] = begin_state['obj_pos']
        self.data.site_xpos[self.S_grasp_sid] = begin_state['palm_pos']
        self.data.site_xpos[self.tool_sid] = begin_state['tool_pos']
        self.data.site_xpos[self.goal_sid] = begin_state['goal_pos']
        self.init_state['init_obj_pos'] = begin_state['obj_pos']
        self.init_state['init_target_pos'] = begin_state['target_pos']
        self.sim.forward()
        return self.get_obs()

    def load_data(self, demo_starting_state, demo_terminal_state):
        self.demo_starting_state = demo_starting_state
        self.demo_terminal_state = demo_terminal_state
        self.demo_num = len(demo_starting_state)

    def get_target_rl_goal(self):
        terminal_state = self.demo_terminal_state[self.random_index]
        target_palm_pos = terminal_state['palm_pos']
        return target_palm_pos

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id('nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel().copy()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos, obj_pos=obj_pos,
                    palm_pos=palm_pos, tool_pos=tool_pos, goal_pos=goal_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:  # nail insude board for 25 steps
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage

    def render(self, mode='human'):
        self.mj_render()

    def reset(self):
        return self._reset()
