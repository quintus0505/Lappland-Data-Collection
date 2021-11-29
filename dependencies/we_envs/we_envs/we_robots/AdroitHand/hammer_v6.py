import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
from we_envs.we_robots.AdroitHand.utils.quatmath import *
import os
import math

ADD_BONUS_REWARDS = True
import copy


class HammerEnvV6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        self.use_state_same_dim = False
        self.goal_achieved_threshold = 0.02

        self.scale = 1.0
        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.goal = dict()
        self.goal['HammerApproachTool'] = np.zeros(3, dtype=np.float)
        self.goal['HammerApproachNail'] = np.zeros(3, dtype=np.float)
        self.goal['HammerNailGoInside'] = np.zeros(3, dtype=np.float)

        self.primitive_name = ''
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitives_goal_achieved_reward = 5.0  # the reward when current primitive's goal is achieved
        self.task_goal_achieved_reward = 5.0  # the reward when total task goal is achieved

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

        self.hand_base_sid = self.sim.model.site_name2id('hand_base')
        self.hand_base_bid = self.sim.model.body_name2id('forearm')

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

        self.inital_board_x = self.model.body_pos[self.model.body_name2id('nail_board'), 0]
        self.inital_board_y = self.model.body_pos[self.model.body_name2id('nail_board'), 1] # is 0.0

    def set_primitive_name(self, primitive_name):
        self.primitive_name = primitive_name

    def set_state_same_dim(self, use_state_same_dim):
        self.use_state_same_dim = use_state_same_dim
        # when the env-observation-dim is changed, following function must be called to get the new correct dim.
        self.get_env_dim()

    def set_goal(self, HammerApproachTool_goal, HammerApproachNail_goal, HammerNailGoInside_goal):
        self.goal['HammerApproachTool'] = np.array(HammerApproachTool_goal, dtype=np.float)
        self.goal['HammerApproachNail'] = np.array(HammerApproachNail_goal, dtype=np.float)
        self.goal['HammerNailGoInside'] = np.array(HammerNailGoInside_goal, dtype=np.float)

    def set_primitive_goal(self, primitive_goal):
        if self.primitive_name == 'HammerApproachTool':
            self.goal['HammerApproachTool'] = primitive_goal
        elif self.primitive_name == 'HammerApproachNail':
            self.goal['HammerApproachNail'] = primitive_goal
        elif self.primitive_name == 'HammerNailGoInside':
            self.goal['HammerNailGoInside'] = primitive_goal

    def enter_condition(self, primitive_name):
        assert primitive_name in ['HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside']
        full_state = self.get_env_state()
        return True

    def leave_condition(self, primitive_name):
        assert primitive_name in ['HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside']
        full_state = self.get_env_state()
        qpos, qvel, board_pos, target_pos, palm_pos, obj_pos = full_state['qpos'], full_state['qvel'], full_state[
            'board_pos'], full_state['target_pos'], full_state['palm_pos'], full_state['obj_pos']
        tool_pos, goal_pos = full_state['tool_pos'], full_state['goal_pos']

        if primitive_name == 'HammerApproachTool':
            if np.linalg.norm(palm_pos - obj_pos) <= 0.08:
                return True
            else:
                return False
        elif primitive_name == 'HammerApproachNail':
            if np.linalg.norm(palm_pos - obj_pos) < 0.085 and np.linalg.norm(obj_pos - self.init_state['obj_pos']) >= 0.08:
                # if np.linalg.norm(tool_pos - target_pos) <= 0.06:
                return True
            else:
                return False
        elif primitive_name == 'HammerNailGoInside':
            if np.linalg.norm(target_pos - goal_pos) <= 0.01 and np.linalg.norm(palm_pos - obj_pos) < 0.085:
                return True
            else:
                return False

    def get_origianl_step_reward(self):
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()

        # get to hammer
        reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
        # take hammer head to nail
        reward -= 0.4*np.linalg.norm((tool_pos - target_pos))
        # make nail go inside
        reward -= 5 * np.linalg.norm(target_pos - goal_pos)
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())
        # print([- 0.1 * np.linalg.norm(palm_pos - obj_pos), -np.linalg.norm((tool_pos - target_pos)),
        #        -10 * np.linalg.norm(target_pos - goal_pos),-1e-2 * np.linalg.norm(self.data.qvel.ravel())])

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 0.2

            # bonus for hammering the nail
            # if (np.linalg.norm(target_pos - goal_pos) < 0.020):
            #     reward += 25
            if (np.linalg.norm(target_pos - goal_pos) < self.goal_achieved_threshold):
                reward += 100
        return reward

    def step_original(self, a):
        a = np.clip(a, -1.0, 1.0)
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

        reward = self.get_origianl_step_reward()

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < self.goal_achieved_threshold else False

        return ob, reward, goal_achieved, dict(goal_achieved=goal_achieved)

    def step(self, a):
        # if not use primitive, return the original version
        if self.primitive_name == '':
            return self.step_original(a=a)

        a = np.clip(a, -1.0, 1.0)
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

        reward_total = reward = 0.0
        reward_tb = np.zeros(5, dtype=np.float)
        assert not self.goal is None, "please set the goal-of-primitive for envirnment first"  # TODO: useless at present
        current_primitives_goal_achieved = False

        if self.primitive_name == 'HammerApproachTool':
            # get to hammer
            reward -= 0.1 * np.linalg.norm(palm_pos - obj_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

            current_primitives_goal_achieved = self.leave_condition(primitive_name='HammerApproachTool')
            self.primitives_goal_achieved[0] = current_primitives_goal_achieved
            if current_primitives_goal_achieved:
                reward += 2.0
        elif self.primitive_name == 'HammerApproachNail':
            # approach the  hammer
            if np.linalg.norm(palm_pos - obj_pos) >= 0.06:
                reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())
            # pick the  hammer
            if np.linalg.norm(obj_pos - self.init_state['obj_pos']) < 0.1:
                reward -= (0.1 - np.linalg.norm(obj_pos - self.init_state['obj_pos']))

            current_primitives_goal_achieved = self.leave_condition(primitive_name='HammerApproachNail')
            self.primitives_goal_achieved[1] = current_primitives_goal_achieved
            if current_primitives_goal_achieved:
                reward += 2.0
        elif self.primitive_name == 'HammerNailGoInside':
            # hold the hammer
            if np.linalg.norm(palm_pos - obj_pos) >= 0.06:
                reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())
            # make nail go inside
            reward -= 10 * np.linalg.norm(target_pos - goal_pos)

            current_primitives_goal_achieved = self.leave_condition(primitive_name='HammerNailGoInside')
            self.primitives_goal_achieved[2] = current_primitives_goal_achieved
            if current_primitives_goal_achieved:
                reward += 2.0

        task_goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < self.goal_achieved_threshold else False
        reward_total = reward
        if task_goal_achieved:
            reward_total += self.task_goal_achieved_reward

        return ob, reward_total, task_goal_achieved, dict(goal_achieved=task_goal_achieved,
                                                          primitives_goal_achieved=self.primitives_goal_achieved,
                                                          current_primitives_goal_achieved=current_primitives_goal_achieved)

    def get_obs(self):
        if self.use_state_same_dim:
            return self.get_obs_same_dim()
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

    def get_obs_same_dim(self):
        qp = self.data.qpos.ravel().copy()
        hand_qpos = qp[2:26]  # 2 wrist angles+ 22 finger angles
        hand_base_pos = self.data.site_xpos[self.hand_base_sid].ravel().copy()
        hand_base_euler = quat2euler(self.data.body_xquat[self.hand_base_bid]).ravel().copy()

        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_euler = quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()

        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_pos_xy = target_pos[0:2]
        nail_impact = np.clip(self.sim.data.sensordata[self.sim.model.sensor_name2id('S_nail')], -1.0, 1.0)

        # return np.concatenate([hand_base_pos, hand_base_euler, hand_qpos, obj_pos, obj_euler, nail_pos_xy,
        #                        np.array([nail_impact])])  # 3+3+24+3+3+1+1+1
        return np.concatenate([hand_base_pos, hand_base_euler, hand_qpos, obj_pos, obj_euler, target_pos])  # 3+3+24+3+3+1+1+1

    def get_images(self):
        if self.viewer is None:
            self.mj_viewer_setup()
        image_data_left = self.sim.render(width=512, height=512, camera_name='camera_left', depth=False)
        image_data_right = self.sim.render(width=512, height=512, camera_name='camera_right', depth=False)
        return [image_data_left, image_data_right]

    def get_images_traingle_cameras(self):
        if self.viewer is None:
            self.mj_viewer_setup()
        image_data_left = self.sim.render(width=512, height=512, camera_name='camera_left', depth=False)
        image_data_right = self.sim.render(width=512, height=512, camera_name='camera_right', depth=False)
        image_data_middle = self.sim.render(width=512, height=512, camera_name='camera_middle', depth=False)
        return [image_data_left, image_data_right, image_data_middle]

    @staticmethod
    def extract_goal_from_obs(primitive_name, obs):  # extract goal from original-obs,
        if primitive_name == 'HammerApproachTool':  # goal is *obj_pos*
            return obs[-10:-7]
        elif primitive_name == 'HammerApproachNail':  # goal is *obj_pos*
            return obs[-10:-7]
        elif primitive_name == 'HammerNailGoInside':  # goal is *nail_pos_xy, [nail_impact]*
            # return obs[-4:-2] + obs[-1]  #target_pos x&y, and nail impact
            return obs[-4:-1] #target_pos, without nail_impact
        else:
            print('primitive_name is not correct')

    @staticmethod
    def extract_goal_from_obs_same_dim(primitive_name, obs):  # extract goal from same-dim-obs
        if primitive_name == 'HammerApproachTool':  # goal is *obj_pos*
            return obs[-9:-6]
        elif primitive_name == 'HammerApproachNail':  # goal is *obj_pos*
            return obs[-9:-6]
        elif primitive_name == 'HammerNailGoInside':  # goal is *nail_pos_xy, [nail_impact]*
            # return obs[-3:] #target_pos x&y, and nail impact
            return obs[-3:]
        else:
            print('primitive_name is not correct')

    def set_noise_scale(self, scale):
        self.scale = scale

    def reset_primtive_env(self, begin_state):
        qp = begin_state['qpos']
        qv = begin_state['qvel']
        self.set_state(qp, qv)

        self.model.body_pos[self.model.body_name2id('nail_board')] = begin_state['board_pos']
        self.data.site_xpos[self.target_obj_sid] = begin_state['target_pos']
        self.data.body_xpos[self.obj_bid] = begin_state['obj_pos']
        self.data.site_xpos[self.S_grasp_sid] = begin_state['palm_pos']
        self.data.site_xpos[self.tool_sid] = begin_state['tool_pos']
        self.data.site_xpos[self.goal_sid] = begin_state['goal_pos']



        for _ in range(500):
            self.sim.forward()

        self.init_state = self.get_env_state()
        return self.get_obs()

    def reset_model(self):
        self.sim.reset()
        target_bid = self.model.body_name2id('nail_board')
        # self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=0.1 * self.scale, high=0.25 * self.scale)
        self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=max(0.1, 0.175-0.075 * self.scale), high=0.175+0.075 * self.scale)
        self.sim.forward()
        self.init_state = self.get_env_state()
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitive_name=''
        return self.get_obs()

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
        nail_impact = np.array([np.clip(self.sim.data.sensordata[self.sim.model.sensor_name2id('S_nail')], -1.0, 1.0)])
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos, obj_pos=obj_pos,
                    palm_pos=palm_pos, tool_pos=tool_pos, goal_pos=goal_pos, nail_impact=nail_impact)

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
        self.init_state = self.get_env_state()

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

    def render(self, mode='human', extra_info=None):
        self.mj_render()
        self.render_text(name='Current Primitive', text=self.primitive_name, location='top_left')
        if extra_info is not None:
            for key in extra_info.keys():
                value = extra_info[key]
                self.render_text(name=key, text=str(value), location='top_left', interval=0)

    def reset(self):
        return self._reset()


    def reset_for_DAPG_policy_train(self, random_noise_level, scale_range=[1,1]):
        """
        different noise area learning and curriculum learning
        :param random_noise_level: noise level area
        :param scale_range:  for curriculum learning, value <= 1
        [1,1] means normal noise level range ,[0.1, 0.1] is narrowest range
        :return:
        """

        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.reset_model()

        board_z_noise_level = int(random_noise_level[0])
        board_y_noise_level = int(random_noise_level[1])

        board_z_noise=[[0.06, 0.08],[0.08, 0.10],[0.10, 0.12],[0.12, 0.14],[0.14, 0.16],[0.16, 0.18],[0.18, 0.20],[0.20, 0.22],[0.22, 0.24],
                       [0.24, 0.26],
                       [0.26, 0.28],[0.28, 0.30],[0.30, 0.32],[0.32, 0.34],[0.34, 0.36],[0.36, 0.38],[0.38, 0.40],[0.40, 0.42],[0.42, 0.44]]

        y = copy.deepcopy(self.inital_board_y)
        board_y_noise=[[y-0.011, y-0.009],[y-0.010, y-0.008],[y-0.009, y-0.007],[y-0.008, y-0.006],[y-0.007, y-0.005],
                       [y-0.006, y-0.004],[y-0.005, y-0.003],[y-0.004, y-0.002],[y-0.003, y-0.001],
                      [y-0.001, y+0.001],
                       [y+0.001, y+0.003],[y+0.002, y+0.004],[y+0.003, y+0.005],[y+0.004, y+0.006],[y+0.005, y+0.007],[y+0.006, y+0.008],[y+0.007, y+0.009],
                       [y+0.008, y+0.010],[y+0.009, y+0.011]]


        assert board_z_noise_level <len(board_z_noise) and  board_z_noise_level >=0 and board_y_noise_level>=0 and board_y_noise_level<len(board_y_noise)
        if scale_range == []:
            scale_range = [1,1]
        # print(board_z_noise[board_z_noise_level])
        # print(board_y_noise[board_y_noise_level])
        # print('---------------------------')
        def set_scale_bound(original_low, original_high, scale):
            assert scale>0, "noise scale is not correct!"
            if scale<0.1:
                scale = 0.1
                print('At present noise scale should not below 0.1, maybe set noise range too small?')
            scaled_range = (original_high- original_low)*(scale-1)/2
            new_low = original_low-scaled_range
            new_high = original_high+scaled_range
            return [new_low, new_high]

        z_low , z_high = board_z_noise[board_z_noise_level][0], board_z_noise[board_z_noise_level][1]
        y_low,  y_high = board_y_noise[board_y_noise_level][0], board_y_noise[board_y_noise_level][1]


        target_bid = self.model.body_name2id('nail_board')
        # self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=max(0.1, 0.175-0.075 * self.scale), high=0.175+0.075 * self.scale)

        self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=set_scale_bound(z_low, z_high, scale_range[0])[0],
                                                                       high=set_scale_bound(z_low, z_high, scale_range[0])[1])
        self.model.body_pos[target_bid, 1] = self.np_random.uniform(low=set_scale_bound(y_low, y_high, scale_range[1])[0],
                                                                       high=set_scale_bound(y_low, y_high, scale_range[1])[1])

        for _ in range(50):
            self.sim.forward()
        self.init_state = self.get_env_state()
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitive_name=''
        return self.get_obs()

