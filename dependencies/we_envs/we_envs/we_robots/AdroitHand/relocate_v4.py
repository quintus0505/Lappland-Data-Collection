import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
import random

ADD_BONUS_REWARDS = True

class RelocateEnvV4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.scale = 1.0

        self.init_state = dict()
        self.init_state_for_terminial_use = dict()
        self.human_set_goal  = None
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
        self.flag = 1

        self.primitive_name = ' '
        self.primitives_goal_achieved = [0, 0 ,0]



        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/../assets/mj_envs/DAPG_relocate.xml', 5)
        
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
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

    def set_primitive_name(self, primitive_name):
        self.primitive_name = primitive_name

    def set_goal(self, set_target):
        self.human_set_target = set_target
        return 0

    def load_data(self, demo_starting_state, demo_terminal_state):
        self.demo_starting_state=demo_starting_state
        self.demo_terminal_state=demo_terminal_state
        self.demo_num=len(demo_starting_state)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        terminal_obj_pos = self.demo_terminal_state[self.random_index]['obj_pos']
        terminal_palm_pos = self.demo_terminal_state[self.random_index]['palm_pos']
        # print(np.linalg.norm(palm_pos-obj_pos))

        reward_tb = np.zeros(5, dtype=np.float)

        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        hand_qpos = qp[:30]

        reward = -0.5 * np.linalg.norm(palm_pos - terminal_palm_pos)
        '''
        rewards for different primitives
        '''
        if self.primitive_name=='Approach':
            '''
                reward for Approach
            '''
            reward = -0.5*np.linalg.norm(palm_pos-terminal_palm_pos)              # take hand to object / terminal_palm_pos
            reward_tb[0] = reward
            # centre palm faces to object
            reward -= 0.1*abs(qp[5])
            reward_tb[1] = -0.1*abs(qp[5])

            if np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])>0.03:
                self.flag = 0
                reward -= 5.0*np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])
                reward_tb[2] = -5.0*np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])

            # velocity penalty
            # reward -= 0.05 * np.linalg.norm(self.data.qvel.ravel()) if palm_pos[2]<0.5 else 0

            if palm_pos[2]>0.25:
                reward -= palm_pos[2]

            if ADD_BONUS_REWARDS:
                if np.linalg.norm(obj_pos-self.init_state['init_obj_pos']) < 0.05:
                    if np.linalg.norm(palm_pos-terminal_palm_pos) < 0.08:        # palm_pos - obj_pos
                        reward += 5.0
                        reward_tb[3] = 5.0
                    if np.linalg.norm(palm_pos-terminal_palm_pos) < 0.04:
                        reward += 10.0
                        reward_tb[3] = 10.0

            self.primitives_goal_achieved[0] = True if (np.linalg.norm(palm_pos-obj_pos) < 0.08 and np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])<0.05 )else False

        elif self.primitive_name=='Grasp':
            '''
                reward for Grasp
            '''
            reward = -1.0*np.linalg.norm(palm_pos-obj_pos)      # make hand go to target
            reward_tb[0] = reward

            if np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])>0.1:
                reward -= 2.0*np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])
                reward_tb[1] = -2.0*np.linalg.norm(obj_pos-self.init_state['init_obj_pos'])

            reward += -1.0 * np.linalg.norm(palm_pos - terminal_palm_pos)  # make hand go to target
            reward_tb[2] = -1.0 * np.linalg.norm(palm_pos - terminal_palm_pos)
            reward += -1.0 * np.linalg.norm(obj_pos -terminal_obj_pos)  # make object go to target
            reward_tb[3] = -1.0 * np.linalg.norm(obj_pos - terminal_obj_pos)
            reward -= 1.0*palm_pos[2]

            if np.linalg.norm(obj_pos - terminal_obj_pos)<0.05:
                reward += 5.0
                reward_tb[4] = 5.0

            self.primitives_goal_achieved[1] = True if (np.linalg.norm(palm_pos-obj_pos) < 0.05 and np.linalg.norm(palm_pos - terminal_palm_pos)<0.05 and np.linalg.norm(obj_pos - terminal_obj_pos)<0.05 )else False

        elif self.primitive_name=='Move2Target':
            '''
                reward for Move2Target
            '''
            # reward = -5.0 * np.linalg.norm(palm_pos - obj_pos)  # make hand go to obj
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
                    reward_tb[4] = 4.0         
            self.primitives_goal_achieved[2] = True if (np.linalg.norm(palm_pos - obj_pos) < 0.05  and np.linalg.norm(
            obj_pos - target_pos) < 0.06) else False

        goal_achieved = np.linalg.norm(obj_pos - target_pos) < 0.075

        return ob, reward, False, dict(goal_achieved=goal_achieved,hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv, rewardtb=reward_tb, primitives_goal_achieved = self.primitives_goal_achieved)



    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel().copy()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])

    def obj2palm(self):
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        return np.linalg.norm(obj_pos-palm_pos)

    def obj2target(self):
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        return np.linalg.norm(target_pos-obj_pos)

    def obj_moved_distance(self):
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        init_obj_pos = self.init_state['init_obj_pos']
        return np.linalg.norm(obj_pos-init_obj_pos)

    def set_noise_scale(self, scale):
        self.scale = scale
       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15*self.scale, high=0.15*self.scale)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15*self.scale, high=0.3*self.scale)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2*self.scale, high=0.2*self.scale)
        self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2*self.scale, high=0.2*self.scale)
        self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15*self.scale, high=0.35*self.scale)

        self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()

        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid,0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid,0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv
        self.flag = 1
        self.primitives_goal_achieved = [0, 0 ,0]

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
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel().copy()
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
            if np.sum(path['env_infos']['goal_achieved']) > 25:     # object close to target for 25 steps
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def render(self, mode='human'):
        self.mj_render()
        self.render_text(name='Current Primitive', text='haha', location='top_right')

    def reset(self):
        return self._reset()
