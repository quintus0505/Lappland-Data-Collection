import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
import random
from we_envs.we_robots.we_utils import rotations
ADD_BONUS_REWARDS = True

"""
change the obs of Env. previous env's observation is fucking shit
"""


class RelocateEnvV6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.scale = 1.0

        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.goal = dict()
        self.goal['Approach'] = np.zeros(3, dtype=np.float)
        self.goal['Grasp'] = np.zeros(3, dtype=np.float)
        self.goal['Move2Target'] = np.zeros(3, dtype=np.float)

        self.goal_limit = dict()
        self.goal_limit['Approach'] = np.array([[-0.3, -0.3, 0], [0.3, 0.3, 0.5]], dtype=np.float)
        self.goal_limit['Grasp'] = np.array([[-0.5, -0.5, 0], [0.5, 0.5, 0.8]], dtype=np.float)
        self.goal_limit['Move2Target'] = np.array([[-0.5, -0.5, 0], [0.5, 0.5, 1]], dtype=np.float)

        self.init_state['init_obj_pos'] = np.array([0, 0, 0])
        self.init_state['init_target_obj_pos'] = np.array([0, 0, 0])

        self.primitive_name = ''
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitives_goal_achieved_reward = 5.0  # the reward when current primitive's goal is achieved
        self.task_goal_achieved_reward = 50.0  # the reward when total task goal is achieved

        self.use_state_same_dim = False
        self.goal_achieved_threshold = 0.04

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
        self.hand_base_sid = self.sim.model.site_name2id('hand_base')
        self.hand_base_bid = self.sim.model.body_name2id('forearm')

        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

    def set_state_same_dim(self, use_state_same_dim):
        self.use_state_same_dim = use_state_same_dim
        # when the env-observation-dim is changed, following function must be called to get the new correct dim.
        self.get_env_dim()

    def set_primitive_name(self, primitive_name):
        self.primitive_name = primitive_name

    def set_goal(self, approach_goal, grasp_goal, move2target_goal):
        self.goal['Approach'] = np.array(approach_goal, dtype=np.float)
        self.goal['Grasp'] = np.array(grasp_goal, dtype=np.float)
        self.goal['Move2Target'] = np.array(move2target_goal, dtype=np.float)

    def set_primitive_goal(self, primitive_goal):

        if self.primitive_name == 'Approach':
            self.goal['Approach'] = primitive_goal
        elif self.primitive_name == 'Grasp':
            self.goal['Grasp'] = primitive_goal
        elif self.primitive_name == 'Move2Target':
            self.goal['Move2Target'] = primitive_goal

    def enter_condition(self, primitive_name):
        """
        hand-coded segementation for primitive enter condition
        :param primitive_name: action primitive to be checked
        :return:
        """
        assert primitive_name in ['Approach', 'Grasp', 'Move2Target']
        full_state = self.get_env_state()
        hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
            'target_pos'], full_state['palm_pos']
        qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']

        if primitive_name == 'Approach':
            if np.linalg.norm(obj_pos - palm_pos) > 0.15:
                return True
            else:
                return False
        elif primitive_name == 'Grasp':
            if obj_pos[2] < 0.04 and np.linalg.norm(palm_pos - obj_pos) < 0.10:
                return True
            else:
                return False
        elif primitive_name == 'Move2Target':
            if obj_pos[2] >= 0.04 and np.linalg.norm(palm_pos - obj_pos) < 0.05:
                return True
            else:
                return False

    def leave_condition(self, primitive_name):
        """
        hand-coded segementation for primitive leave condition
        :param primitive_name: action primitive to be checked
        :return:
        """
        assert primitive_name in ['Approach', 'Grasp', 'Move2Target']
        full_state = self.get_env_state()
        hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
            'target_pos'], full_state['palm_pos']
        qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']

        if primitive_name == 'Approach':
            if np.linalg.norm(obj_pos - palm_pos) <= 0.08:
                return True
            else:
                return False
        elif primitive_name == 'Grasp':
            if obj_pos[2] >= 0.04 and np.linalg.norm(palm_pos - obj_pos) < 0.05:
                return True
            else:
                return False
        elif primitive_name == 'Move2Target':
            if np.linalg.norm(obj_pos - target_pos) <= 0.04:
                return True
            else:
                return False

    def get_origianl_step_reward(self):
        ob = self.get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 0.1                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target

        reward -= 0.1
        if ADD_BONUS_REWARDS:
            # if np.linalg.norm(obj_pos-target_pos) < 0.1:
            #     reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < self.goal_achieved_threshold*1.2:
                reward += 1.0                                          # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < self.goal_achieved_threshold else False
        if goal_achieved:
            reward += self.task_goal_achieved_reward
        # print(reward)
        return reward

    def step_original(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        # reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        # if obj_pos[2] > 0.04:                                       # if object off the table
        #     reward += 0.1                                           # bonus for lifting the object
        #     reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
        #     reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target
        #
        # reward -= 0.1
        # if ADD_BONUS_REWARDS:
        #     # if np.linalg.norm(obj_pos-target_pos) < 0.1:
        #     #     reward += 10.0                                          # bonus for object close to target
        #     if np.linalg.norm(obj_pos-target_pos) < 0.05:
        #         reward += 1.0                                          # bonus for object "very" close to target
        #
        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < self.goal_achieved_threshold else False
        # if goal_achieved:
        #     reward += self.task_goal_achieved_reward
        reward = self.get_origianl_step_reward()

        return ob, reward, goal_achieved, dict(goal_achieved=goal_achieved)

    def step(self, a):
        # if not use primitive, return the original version
        if self.primitive_name=='':
            return self.step_original(a=a)

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

        # print(np.linalg.norm(palm_pos-obj_pos))

        reward_tb = np.zeros(5, dtype=np.float)

        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        hand_qpos = qp[:30]

        reward_total = 0.0
        assert not self.goal is None, "please set the goal-of-primitive for envirnment first"

        current_primitives_goal_achieved = False

        if self.primitive_name == 'Approach':
            '''
                reward for Approach
            '''
            reward_tb[0] = -0.5 * np.linalg.norm(
                palm_pos - self.goal['Approach'])  # take hand to object / terminal_palm_pos
            reward_total += reward_tb[0]
            # centre palm faces to object
            reward_tb[1] = -0.1 * abs(qp[5])
            reward_total += reward_tb[1]

            if np.linalg.norm(obj_pos - self.init_state['init_obj_pos']) > 0.03:  # the object moved by hand accidently

                reward_tb[2] = -5.0 * np.linalg.norm(obj_pos - self.init_state['init_obj_pos'])
                reward_total += reward_tb[2]

            # velocity penalty
            # reward -= 0.05 * np.linalg.norm(self.data.qvel.ravel()) if palm_pos[2]<0.5 else 0

            if palm_pos[2] > 0.25:
                reward_total -= palm_pos[2]

            current_primitives_goal_achieved = self.leave_condition(primitive_name='Approach')
            self.primitives_goal_achieved[0] = current_primitives_goal_achieved

            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[0]:
                    reward_tb[3] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[3]

        elif self.primitive_name == 'Grasp':
            '''
                reward for Grasp
            '''
            reward_tb[0] = -1.0 * np.linalg.norm(palm_pos - obj_pos)  # make hand go to target
            reward_total += reward_tb[0]

            if np.linalg.norm(obj_pos - self.init_state['init_obj_pos']) > 0.1:
                reward_tb[1] -= 2.0 * np.linalg.norm(obj_pos - self.init_state['init_obj_pos'])
                reward_total += reward_tb[1]

            reward_tb[2] = -1.0 * np.linalg.norm(palm_pos - self.goal['Grasp'])  # make hand go to target
            reward_total += reward_tb[2]
            reward_tb[3] = -1.0 * np.linalg.norm(obj_pos - self.goal['Grasp'])  # make object go to target
            reward_total += reward_tb[3]

            # current_primitives_goal_achieved = True if np.linalg.norm(obj_pos - self.init_state['init_obj_pos']) >= 0.03 \
            #                                            and np.linalg.norm(palm_pos - obj_pos) < 0.05 else False
            current_primitives_goal_achieved = self.leave_condition(primitive_name='Grasp')
            self.primitives_goal_achieved[1] = current_primitives_goal_achieved

            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[1]:
                    reward_tb[4] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[4]

        elif self.primitive_name == 'Move2Target':
            '''
                reward for Move2Target
            '''
            reward_tb[0] = -5.0 * np.linalg.norm(palm_pos - obj_pos)  # make hand go to obj
            reward_total += reward_tb[0]

            if obj_pos[2] > 0.04:
                reward_tb[1] = 1.0
                reward_total += reward_tb[1]

                reward_tb[3] = -1.0 * np.linalg.norm(palm_pos - self.goal['Move2Target'])
                reward_total += reward_tb[3]

            # current_primitives_goal_achieved = True if (np.linalg.norm(obj_pos - target_pos) < 0.04) else False
            current_primitives_goal_achieved = self.leave_condition(primitive_name='Move2Target')
            self.primitives_goal_achieved[2] = current_primitives_goal_achieved

            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[2]:
                    reward_tb[4] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[4]

        task_goal_achieved = np.linalg.norm(obj_pos - target_pos) < self.goal_achieved_threshold

        if task_goal_achieved:
            reward_total += self.task_goal_achieved_reward
            # print("task finished!")

        # print(reward_total)
        return ob, reward_total, task_goal_achieved, dict(goal_achieved=task_goal_achieved, hand_qpos=hand_qpos,
                                                          obj_pos=obj_pos,
                                                          target_pos=target_pos, palm_pos=palm_pos,
                                                          qpos=qp, qvel=qv, rewardtb=reward_tb,
                                                          primitives_goal_achieved=self.primitives_goal_achieved,
                                                          current_primitives_goal_achieved=current_primitives_goal_achieved)

    def get_obs(self):
        if self.use_state_same_dim:
            return self.get_obs_same_dim()
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel().copy()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return np.concatenate([qp[:-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos])

    def get_obs_deprecated(self):
        qp = self.data.qpos.ravel().copy()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()

        hand_qpos = qp[:30]

        return np.concatenate([hand_qpos, palm_pos, obj_pos, target_pos])

    @staticmethod
    def extract_goal_from_obs(primitive_name, obs): # extract goal from original-obs
        if primitive_name=='Approach': # goal is *obj_pos*
            return obs[-6:-3]
        elif primitive_name=='Grasp': # goal is *obj_pos*
            return obs[-6:-3]
        elif primitive_name=='Move2Target': # goal is *target_pos*
            return obs[-3:]
        else:
            print('primitive_name is not correct')

    @staticmethod
    def extract_goal_from_obs_same_dim(primitive_name, obs): # extract goal from same-dim-obs
        if primitive_name=='Approach': # goal is *obj_pos*
            return obs[-9:-6]
        elif primitive_name=='Grasp': # goal is *obj_pos*
            return obs[-9:-6]
        elif primitive_name=='Move2Target': # goal is *target_pos*
            return obs[-3:]
        else:
            print('primitive_name is not correct')

    def get_obs_same_dim(self):
        qp = self.data.qpos.ravel().copy()
        hand_qpos = qp[6:30] # 2 wrist angles+ 22 finger angles

        hand_base_pos = self.data.site_xpos[self.hand_base_sid].ravel().copy()
        # hand_base_r = rotations.mat2euler(self.data.site_xmat[self.hand_base_sid])
        hand_base_euler = rotations.quat2euler(self.data.body_xquat[self.hand_base_bid]).ravel().copy()

        obj_pos = self.data.body_xpos[self.obj_bid].ravel().copy()
        obj_euler = rotations.quat2euler(self.data.body_xquat[self.obj_bid]).ravel().copy()

        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        # return {'hand_qpos':hand_qpos, 'hand_base_t':hand_base_t, 'hand_base_r':hand_base_r, 'obj_pos':obj_pos, 'target_pos':target_pos}
        return np.concatenate([hand_base_pos, hand_base_euler, hand_qpos, obj_pos, obj_euler, target_pos]) # 3+3+24+3+3+3

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

    def set_noise_scale(self, scale):
        self.scale = scale

    def reset_primtive_env(self, begin_state):
        qp = begin_state['qpos']
        qv = begin_state['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = begin_state['obj_pos']
        self.model.site_pos[self.target_obj_sid] = begin_state['target_pos']
        self.data.site_xpos[self.S_grasp_sid] = begin_state['palm_pos']

        for _ in range(500):
            self.sim.forward()

        # for _ in range(500):
        #     self.sim.step()

        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid, 0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid, 0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv

        return self.get_obs()

    def reset_model(self):
        def set_scale_bound(original_low, original_high, scale):
            assert scale>0, "noise scale is not correct!"
            if scale<0.1:
                scale = 0.1
                print('At present noise scale should not below 0.1, maybe set noise range too small?')
            scaled_range = (original_high- original_low)*(scale-1)/2
            new_low = original_low-scaled_range
            new_high = original_high+scaled_range
            return [new_low, new_high]

        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=set_scale_bound(-0.15, 0.15, self.scale)[0],
                                                                      high=set_scale_bound(-0.15, 0.15, self.scale)[1])

        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=set_scale_bound(-0.15, 0.3, self.scale)[0],
                                                                      high=set_scale_bound(-0.15, 0.3, self.scale)[1])

        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=set_scale_bound(-0.2, 0.2, self.scale)[0],
                                                                             high=set_scale_bound(-0.2, 0.2, self.scale)[1])

        self.model.site_pos[self.target_obj_sid, 1] = self.np_random.uniform(low=set_scale_bound(-0.2, 0.2, self.scale)[0],
                                                                             high=set_scale_bound(-0.2, 0.2, self.scale)[1])

        self.model.site_pos[self.target_obj_sid, 2] = self.np_random.uniform(low=set_scale_bound(0.15, 0.35, self.scale)[0],
                                                                             high=set_scale_bound(0.15, 0.35, self.scale)[1])

        # self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=-0.15 * self.scale, high=0.15 * self.scale)
        # self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=-0.15 * self.scale, high=0.3 * self.scale)
        # self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2 * self.scale,high=0.2 * self.scale)
        # self.model.site_pos[self.target_obj_sid, 1] = self.np_random.uniform(low=-0.2 * self.scale,high=0.2 * self.scale)
        # self.model.site_pos[self.target_obj_sid, 2] = self.np_random.uniform(low=0.15 * self.scale,high=0.35 * self.scale)

        for _ in range(500):
            self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()

        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid, 0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid, 0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv

        self.primitives_goal_achieved = [0, 0, 0]
        self.primitive_name=''

        return self.get_obs()

    def reset_for_DAPG_policy_train(self, random_noise_level, scale_range=[1,1]):
        """
        different noise area learning and curriculum learning
        :param random_noise_level: noise level area
        :param scale_range:  for curriculum learning, value <= 1
        [1,1] means normal noise level range ,[0.1, 0.1] is narrowest range
        :return:
        """
        self.sim.reset()
        self.sim.forward()

        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()

        self.reset_model()
        obj_x_noise_level = int(random_noise_level[0])
        obj_y_noise_level = int(random_noise_level[1])
        # obj_x_noise=[[-0.75, -0.6], [-0.6, -0.45], [-0.45, -0.3], [-0.3, -0.15], [-0.15,0],
        #              [0,0.15], [0.15,0.3],[0.3,0.45],[0.45,0.6],[0.6,0.75]]
        # obj_y_noise=[[-0.75, -0.6], [-0.6, -0.45], [-0.45, -0.3], [-0.3, -0.15], [-0.15,0],
        #              [0,0.15], [0.15,0.3],[0.3,0.45],[0.45,0.6],[0.6,0.75]]

        obj_x_noise=[[-0.475, -0.425],[-0.425, -0.375],[-0.375, -0.325],[-0.325, -0.275],[-0.275, -0.225], [-0.225, -0.175],[-0.175, -0.125], [-0.125, -0.075], [-0.075, -0.025],
                     [-0.025,0.025],
                     [0.025,0.075], [0.075,0.125],[0.125,0.175],[0.175,0.225],[0.225,0.275],[0.275,0.325],[0.325,0.375],[0.375,0.425],[0.425,0.475]]

        obj_y_noise=[[-0.4, -0.35],[-0.35, -0.3],[-0.3, -0.25],[-0.25, -0.2],[-0.2, -0.15],[-0.15, -0.1],[-0.1, -0.05],[-0.05, 0],[0, 0.05],
                     [0.05, 0.1],
                     [0.1, 0.15],[0.15, 0.2],[0.2, 0.25],[0.25, 0.3],[0.3, 0.35],[0.35, 0.4],[0.4, 0.45],[0.45, 0.5],[0.5, 0.55]]


        assert obj_x_noise_level <len(obj_x_noise) and obj_y_noise_level <len(obj_y_noise) and obj_x_noise_level >=0 and obj_y_noise_level >=0
        if scale_range == []:
            scale_range = [1,1]
        # print(obj_x_noise[obj_x_noise_level])
        # print(obj_y_noise[obj_y_noise_level])
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

        x_low , x_high = obj_x_noise[obj_x_noise_level][0], obj_x_noise[obj_x_noise_level][1]
        y_low,  y_high = obj_y_noise[obj_y_noise_level][0], obj_y_noise[obj_y_noise_level][1]
        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=set_scale_bound(x_low, x_high, scale_range[0])[0],
                                                                      high=set_scale_bound(x_low, x_high, scale_range[0])[1])

        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=set_scale_bound(y_low, y_high, scale_range[1])[0],
                                                                      high=set_scale_bound(y_low, y_high, scale_range[1])[1])

        for _ in range(500):
            self.sim.forward()
        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid, 0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid, 0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv

        self.primitives_goal_achieved = [0, 0, 0]
        self.primitive_name=''
        return self.get_obs()

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

    def render(self, mode='human', extra_info=None):
        self.mj_render()
        self.render_text(name='Current Primitive', text=self.primitive_name, location='top_left', interval=2)
        if extra_info is not None:
            for key in extra_info.keys():
                value = extra_info[key]
                self.render_text(name=key, text=str(value), location='top_left', interval=0)


    def reset(self):
        return self._reset()
