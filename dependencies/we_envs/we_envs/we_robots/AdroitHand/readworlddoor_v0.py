import numpy as np
from gym import utils
import os
from we_envs.we_robots.we_utils import rotations
import gym.spaces as spaces
import gym

ADD_BONUS_REWARDS = True

class DoorEnvRealworldV0(gym.Env):
    def __init__(self):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0

        #TODO:
        self.goal_dim = 3
        self.obs_dim = 18
        self.act_dim = 3


        self.use_state_same_dim = True
        self.goal_achieved_threshold = 1.35

        self.scale = 1.0
        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.goal = dict()
        self.goal['DoorApproach'] = np.zeros(3, dtype=np.float)
        self.goal['DoorGraspLatch'] = np.zeros(3, dtype=np.float)
        self.goal['DoorOpen'] = np.zeros(3, dtype=np.float)

        self.primitive_name = ''
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitives_goal_achieved_reward = 5.0  # the reward when current primitive's goal is achieved
        self.task_goal_achieved_reward = 5.0  # the reward when total task goal is achieved

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        low = -1*np.ones(self.act_dim,dtype=np.float)
        high = 1*np.ones(self.act_dim,dtype=np.float)
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        ob = self.reset_model()


    def set_primitive_name(self, primitive_name):
        self.primitive_name = primitive_name

    def set_state_same_dim(self, use_state_same_dim):
        self.use_state_same_dim = use_state_same_dim
        # when the env-observation-dim is changed, following function must be called to get the new correct dim.
        self.get_env_dim()

    def get_env_dim(self):
        pass

    def set_goal(self, DoorApproach_goal, DoorGraspLatch_goal, DoorOpen_goal):
        self.goal['DoorApproach'] = np.array(DoorApproach_goal, dtype=np.float)
        self.goal['DoorGraspLatch'] = np.array(DoorGraspLatch_goal, dtype=np.float)
        self.goal['DoorOpen'] = np.array(DoorOpen_goal, dtype=np.float)

    def set_primitive_goal(self, primitive_goal):
        if self.primitive_name == 'DoorApproach':
            self.goal['DoorApproach'] = primitive_goal
        elif self.primitive_name == 'DoorGraspLatch':
            self.goal['DoorGraspLatch'] = primitive_goal
        elif self.primitive_name == 'DoorOpen':
            self.goal['DoorOpen'] = primitive_goal

    def enter_condition(self, primitive_name):
        assert primitive_name in ['DoorApproach', 'DoorGraspLatch', 'DoorOpen']
        full_state = self.get_env_state()
        return True  # TODO: not used at present

    def leave_condition(self, primitive_name):
        assert primitive_name in ['DoorApproach', 'DoorGraspLatch', 'DoorOpen']
        full_state = self.get_env_state()

        qpos, qvel, door_body_pos, handle_pos, palm_pos = full_state['qpos'], full_state['qvel'], full_state[
            'door_body_pos'], full_state['handle_pos'], full_state['palm_pos']
        latch_pos, door_hinge_pos = full_state['latch_pos'], full_state['door_hinge_pos']


        if primitive_name == 'DoorApproach':
            if np.linalg.norm(handle_pos - palm_pos) <= 0.063:
                return True
            else:
                return False
        elif primitive_name == 'DoorGraspLatch':
            if abs(latch_pos - 1.57) <= 0.3 and abs(door_hinge_pos) >= 0.001:
                return True
            else:
                return False
        elif primitive_name == 'DoorOpen':
            if abs(door_hinge_pos) >= 1.35:
                return True
            else:
                return False

    def get_origianl_step_reward(self):
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)
        # print("{r1},{r2},{r3}".format(r1=-0.1*np.linalg.norm(palm_pos-handle_pos),
        #                               r2=-0.1*(door_pos - 1.57)*(door_pos - 1.57),
        #                               r3=-1e-5*np.sum(self.data.qvel**2)))

        if ADD_BONUS_REWARDS:
            # Bonus
            # if door_pos > 0.2:
            #     reward += 2
            # if door_pos > 1.0:
            #     reward += 8
            if door_pos > self.goal_achieved_threshold:
                reward += 100
        return reward

    def step_original(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        reward = self.get_origianl_step_reward()

        goal_achieved = True if door_pos >= self.goal_achieved_threshold else False

        return ob, reward, goal_achieved, dict(goal_achieved=goal_achieved)

    def step(self, a):
        # if not use primitive, return the original version
        if self.primitive_name == '':
            return self.step_original(a=a)

        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]
        latch_pos = self.data.qpos[-1].ravel()[0]
        door_hinge_pos = self.data.qpos[self.door_hinge_did].ravel().copy()

        reward_total = reward = 0.0
        # reward -= 0.1 #TODO: need every step cost?
        reward_tb = np.zeros(5, dtype=np.float)
        assert not self.goal is None, "please set the goal-of-primitive for envirnment first"  # TODO: useless at present
        current_primitives_goal_achieved = False

        if self.primitive_name == 'DoorApproach':
            # get to handle
            dis_constrain = 0.09
            reward = -0.5*np.linalg.norm(palm_pos-handle_pos)
            if np.linalg.norm(palm_pos-handle_pos)<dis_constrain-0.01:
                reward -= 0.4/np.linalg.norm(palm_pos-handle_pos)

            reward_tb[0] = reward
            if latch_pos>0.05:
                reward -= 1.5**latch_pos
                reward_tb[1]= -1.5**latch_pos

            # velocity cost
            reward += -1e-5*np.sum(self.data.qvel**2)
            reward_tb[2]=-1e-5*np.sum(self.data.qvel**2)

            current_primitives_goal_achieved = self.leave_condition(primitive_name='DoorApproach')
            self.primitives_goal_achieved[0] = current_primitives_goal_achieved
            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[0]:
                    reward_tb[3] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[3]

        elif self.primitive_name == 'DoorGraspLatch':
            constrain = 0.3
            # get  handle
            if np.linalg.norm(palm_pos - handle_pos) > 0.07:
                reward -= 0.5 * np.linalg.norm(palm_pos - handle_pos)
            reward_tb[0] = reward

            # rotate the handle
            reward -= abs(1.57-constrain-latch_pos)
            reward_tb[1] = -abs(1.57-constrain-latch_pos)

            # velocity cost
            reward += -1e-5 * np.sum(self.data.qvel ** 2)
            reward_tb[2] = -1e-5 * np.sum(self.data.qvel ** 2)

            current_primitives_goal_achieved = self.leave_condition(primitive_name='DoorGraspLatch')
            self.primitives_goal_achieved[1] = current_primitives_goal_achieved
            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[1]:
                    reward_tb[3] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[3]

        elif self.primitive_name == 'DoorOpen':
            constrain = 0.05
            # get to handle
            if np.linalg.norm(palm_pos - handle_pos) > 0.08:
                reward -= 0.5 * np.linalg.norm(palm_pos - handle_pos)
            reward_tb[0] = reward
            # make  the door hinge rotate to make door open
            if door_pos<1.57-constrain:
                reward -= abs(1.57-constrain-door_pos)
                reward_tb[1] = -abs(1.57-constrain-door_pos)
            # velocity cost
            reward += -1e-5 * np.sum(self.data.qvel ** 2)
            reward_tb[2] = -1e-5 * np.sum(self.data.qvel ** 2)

            current_primitives_goal_achieved = self.leave_condition(primitive_name='DoorOpen')
            self.primitives_goal_achieved[2] = current_primitives_goal_achieved
            if ADD_BONUS_REWARDS:
                if self.primitives_goal_achieved[2]:
                    reward_tb[3] = self.primitives_goal_achieved_reward
                    reward_total += reward_tb[3]

        task_goal_achieved = True if door_pos >= self.goal_achieved_threshold else False
        reward_total = reward
        # print("primitive: {primitive}, reward: {reward}".format(primitive=self.primitive_name, reward=reward_total))
        if task_goal_achieved:
            reward_total += self.task_goal_achieved_reward

        return ob, reward_total, task_goal_achieved, dict(goal_achieved=task_goal_achieved, rewardtb=reward_tb,
                                                          primitives_goal_achieved=self.primitives_goal_achieved,
                                                          current_primitives_goal_achieved=current_primitives_goal_achieved)



    def get_obs(self):
        # if self.use_state_same_dim:
        return self.get_obs_same_dim()


    def get_obs_same_dim(self):
        current_state = self.get_env_state()
        # hand_base_pos = current_state['hand_base_pos']

        return []
        # return {'hand_qpos':hand_qpos, 'hand_base_t':hand_base_t, 'hand_base_r':hand_base_r, 'obj_pos':obj_pos, 'target_pos':target_pos}
        # return np.concatenate([hand_base_pos, hand_base_euler, hand_qpos, handle_pos, handle_euler, door_pos, [latch_pos], [latch_vel]]) # 3+3+24+3+3+1+1+1



    @staticmethod
    def extract_goal_from_obs(primitive_name, obs): # extract goal from original-obs,
        if primitive_name=='DoorApproach': # goal is *handle_pos*
            return obs[-10:-7]
        elif primitive_name=='DoorGraspLatch': # goal is *handle_pos*
            return obs[-10:-7]
        elif primitive_name=='DoorOpen': # goal is *handle_pos*
            return obs[-10:-7]
        else:
            print('primitive_name is not correct')

    @staticmethod
    def extract_goal_from_obs_same_dim(primitive_name, obs): # extract goal from same-dim-obs
        if primitive_name=='DoorApproach': # goal is *handle_pos*
            return obs[-9:-6]
        elif primitive_name=='DoorGraspLatch': # goal is *handle_pos*
            return obs[-9:-6]
        elif primitive_name=='DoorOpen': # goal is *handle_pos*
            return obs[-9:-6]
        else:
            print('primitive_name is not correct')



    def reset_primtive_env(self, begin_state):
        return self.get_obs()

    def reset_model(self):
        print("please reset the env manually")
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        # qp = self.data.qpos.ravel().copy()
        # qv = self.data.qvel.ravel().copy()
        # door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        # handle_pos = self.data.site_xpos[self.handle_sid].ravel().copy()
        # palm_pos = self.data.site_xpos[self.grasp_sid].ravel().copy()
        # door_hinge_pos = self.data.qpos[self.door_hinge_did].ravel().copy()
        # latch_pos = self.data.qpos[-1] # The latch joint angle
        #TODO: get from robot real control system
        return dict()
        # return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos,
        #             handle_pos=handle_pos, palm_pos=palm_pos, door_hinge_pos=door_hinge_pos, latch_pos=latch_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        pass


    def reset(self):
        return self.reset_model()



