import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
from we_envs.we_robots.we_utils import rotations

ADD_BONUS_REWARDS = True

class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.scale = 1.0

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
        self.hand_base_sid = self.sim.model.site_name2id('hand_base')
        self.hand_base_bid = self.sim.model.body_name2id('forearm')

        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

        self.init_state = dict()

    def step(self, a):
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

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 1.0                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                          # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])

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

       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15*self.scale, high=0.15*self.scale)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15*self.scale, high=0.3*self.scale)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2*self.scale, high=0.2*self.scale)
        self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2*self.scale, high=0.2*self.scale)
        self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15*self.scale, high=0.35*self.scale)

        for _ in range(5000):
            self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()

        self.init_state['init_obj_pos'] = self.model.body_pos[self.obj_bid,0:3]
        self.init_state['init_target_obj_pos'] = self.model.site_pos[self.target_obj_sid,0:3]
        self.init_state['init_qpos'] = qp
        self.init_state['init_qvel'] = qv

        return self.get_obs()

    def set_noise_scale(self, scale):
        self.scale = scale

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

    def reset(self):
        return self._reset()
