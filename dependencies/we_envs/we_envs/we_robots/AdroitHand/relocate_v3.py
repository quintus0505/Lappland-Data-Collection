import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
import random

ADD_BONUS_REWARDS = True


class RelocateEnvV3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0

        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.human_set_goal = None
        self.human_set_target = np.array([0, 0, 0])

        self.init_state['init_obj_pos'] = np.array([0, 0, 0])
        self.init_state['init_target_obj_pos'] = np.array([0, 0, 0])
        self.init_state_for_terminial_use['obj_pos'] = np.array([0, 0, 0])
        self.init_state_for_terminial_use['palm_pos'] = np.array([0, 0, 0])      
        self.demo_starting_state = []
        self.demo_terminal_state = []
        self.demo_terminal_state.append(self.init_state_for_terminial_use)
        self.demo_num = 0
        self.random_index = 0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + '/../assets/mj_envs/DAPG_relocate.xml', 5)

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

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

    def load_data(self, demo_starting_state, demo_terminal_state):
        self.demo_starting_state = demo_starting_state
        self.demo_terminal_state = demo_terminal_state
        self.demo_num = len(demo_starting_state)

    def set_goal(self, set_target):
        self.human_set_target = set_target
        return 0

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        terminal_obj_pos = self.demo_terminal_state[self.random_index]['obj_pos']
        terminal_palm_pos = self.demo_terminal_state[self.random_index]['palm_pos']
        # print(np.linalg.norm(palm_pos-obj_pos))

        reward_tb = np.zeros(5, dtype=np.float)

        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        hand_qpos = qp[:30]
        # new reward
        reward = -5.0 * np.linalg.norm(palm_pos - obj_pos)  # make hand go to obj
        reward_tb[0] = reward

        if obj_pos[2] > 0.04:
            reward += 1.0
            reward_tb[1] = 1.0
            reward -= 1.0 * np.linalg.norm(obj_pos -terminal_obj_pos)  # make object go to target
            reward_tb[2] = -1.0 * np.linalg.norm(obj_pos - terminal_obj_pos)
            reward -= 1.0 * np.linalg.norm(palm_pos - terminal_palm_pos)  # make object go to target
            reward_tb[3] = -1.0 * np.linalg.norm(palm_pos - terminal_palm_pos)

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 2.0
                reward_tb[4] = 2.0
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 4.0
                reward_tb[4] = 4.0                                       # bonus for object "very" close to target

        goal_achieved = True if (np.linalg.norm(palm_pos - obj_pos) < 0.05  and np.linalg.norm(
            obj_pos - target_pos) < 0.1) else False

        return ob, reward, False, dict(goal_achieved=goal_achieved, hand_qpos=hand_qpos, obj_pos=obj_pos,
                                               target_pos=target_pos, palm_pos=palm_pos,
                                               qpos=qp, qvel=qv, rewardtb=reward_tb)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel().copy()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return np.concatenate([qp[:-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos])

    def obj2palm(self):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        return np.linalg.norm(obj_pos - palm_pos)

    def obj2target(self):
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        return np.linalg.norm(target_pos - obj_pos)

    def obj_moved_distance(self):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        init_obj_pos = self.init_state['init_obj_pos']
        return np.linalg.norm(obj_pos - init_obj_pos)

    def reset_model(self):

        '''old reset'''

        '''
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        obj_x = self.model.body_pos[self.obj_bid,0]
        obj_y = self.model.body_pos[self.obj_bid,1] 
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=obj_x-0.1, high=obj_x+0.1)
        self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=obj_y-0.1, high=obj_y+0.1)
        self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.02, high=0.05)
        '''

        self.random_index = random.randint(0, self.demo_num - 1)
        begin_state = self.demo_starting_state[self.random_index]
        terminal_state = self.demo_terminal_state[self.random_index]

        qp = begin_state['qpos']
        qv = begin_state['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = begin_state['obj_pos']
        self.model.site_pos[self.target_obj_sid] = begin_state['target_pos']
        self.data.site_xpos[self.S_grasp_sid] = begin_state['palm_pos']

        self.sim.forward()

        # for _ in range(500):
        #     self.sim.step()

        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid, 0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid, 0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv

        return self.get_obs()

    def get_target_rl_goal(self):
        terminal_state = self.demo_terminal_state[self.random_index]
        target_palm_pos = terminal_state['palm_pos']
        target_obj_pos = terminal_state['obj_pos']
        goal_palm_and_obj_pos = np.append(target_palm_pos, target_obj_pos)
        return target_palm_pos

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
                    qpos=qp, qvel=qv, init_state=self.init_state)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos

        self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:  # object close to target for 25 steps
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage

    def render(self, mode='human'):
        self.mj_render()

    def reset(self):
        return self._reset()
