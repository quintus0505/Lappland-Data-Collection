import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
import random

ADD_BONUS_REWARDS = True


class RelocateEnvV5(mujoco_env.MujocoEnv, utils.EzPickle):
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

        self.init_state['init_obj_pos'] = np.array([0, 0, 0])
        self.init_state['init_target_obj_pos'] = np.array([0, 0, 0])

        self.primitive_name = ''
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitives_goal_achieved_reward = 5.0  # the reward when current primitive's goal is achieved
        self.task_goal_achieved_reward = 5.0  # the reward when total task goal is achieved

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

        # print(np.linalg.norm(palm_pos-obj_pos))

        reward_tb = np.zeros(5, dtype=np.float)

        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        hand_qpos = qp[:30]

        reward_total = 0.0
        assert not self.goal is None, "please set the goal-of-primitive for envirnment first"

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

            # if ADD_BONUS_REWARDS:
            #     if np.linalg.norm(obj_pos - self.init_state['init_obj_pos']) < 0.05:
            #         if np.linalg.norm(palm_pos - self.goal['Approach']) < 0.15:  # palm_pos - obj_pos
            #             reward_total += 5.0
            #             reward_tb[3] = 5.0
            #         if np.linalg.norm(palm_pos - self.goal['Approach']) < 0.12:
            #             reward_total += 10.0
            #             reward_tb[3] = 10.0

            self.primitives_goal_achieved[0] = True if (np.linalg.norm(palm_pos - obj_pos) < 0.15 and np.linalg.norm(
                obj_pos - self.init_state['init_obj_pos']) < 0.05) else False

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

            # if np.linalg.norm(obj_pos - self.goal['Grasp']) < 0.05:
            #     reward_tb[4] = 5.0
            #     reward_total += reward_tb[4]

            # self.primitives_goal_achieved[1] = True if (np.linalg.norm(palm_pos-obj_pos) < 0.05 and
            #                                             np.linalg.norm(palm_pos - self.goal['Grasp'])<0.05 and
            #                                             np.linalg.norm(obj_pos - self.goal['Grasp'])<0.05 ) else False
            self.primitives_goal_achieved[1] = True if np.linalg.norm(obj_pos - self.init_state['init_obj_pos']) >= 0.03 \
                                                       and np.linalg.norm(palm_pos - obj_pos) < 0.05 else False

            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[1]:
                    reward_tb[4] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[4]

        elif self.primitive_name == 'Move2Target':
            '''
                reward for Move2Targt
            '''
            reward_tb[0] = -5.0 * np.linalg.norm(palm_pos - obj_pos)  # make hand go to obj
            reward_total += reward_tb[0]

            if obj_pos[2] > 0.04:
                reward_tb[1] = 1.0
                reward_total += reward_tb[1]

                reward_tb[3] = -1.0 * np.linalg.norm(palm_pos - self.goal['Move2Target'])
                reward_total += reward_tb[3]

            # if ADD_BONUS_REWARDS:
            #     if np.linalg.norm(obj_pos - target_pos) < 0.075:
            #         reward_tb[4] = 10.0
            #         reward_total += reward_tb[4]
            #     if np.linalg.norm(obj_pos - target_pos) < 0.05:
            #         reward_tb[4] = 20.0
            #         reward_total += reward_tb[4]

            self.primitives_goal_achieved[2] = True if (np.linalg.norm(obj_pos - target_pos) < 0.1) else False

            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[2]:
                    reward_tb[4] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[4]

        task_goal_achieved = np.linalg.norm(obj_pos - target_pos) < 0.1
        if task_goal_achieved:
            reward_total += self.task_goal_achieved_reward

        return ob, reward_total, task_goal_achieved, dict(goal_achieved=task_goal_achieved, hand_qpos=hand_qpos, obj_pos=obj_pos,
                                                     target_pos=target_pos, palm_pos=palm_pos,
                                                     qpos=qp, qvel=qv, rewardtb=reward_tb,
                                                     primitives_goal_achieved=self.primitives_goal_achieved)

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

    def set_noise_scale(self, scale):
        self.scale = scale

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=-0.15 * self.scale, high=0.15 * self.scale)
        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=-0.15 * self.scale, high=0.3 * self.scale)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2 * self.scale,
                                                                             high=0.2 * self.scale)
        self.model.site_pos[self.target_obj_sid, 1] = self.np_random.uniform(low=-0.2 * self.scale,
                                                                             high=0.2 * self.scale)
        self.model.site_pos[self.target_obj_sid, 2] = self.np_random.uniform(low=0.15 * self.scale,
                                                                             high=0.35 * self.scale)

        self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()

        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid, 0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid, 0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv

        self.primitives_goal_achieved = [0, 0, 0]

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
        self.render_text(name='Current Primitive', text=self.primitive_name, location='top_right')

    def reset(self):
        return self._reset()
