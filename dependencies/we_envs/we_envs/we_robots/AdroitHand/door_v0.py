import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
from we_envs.we_robots.we_utils import rotations

ADD_BONUS_REWARDS = True

class DoorEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/../assets/mj_envs/DAPG_door.xml', 5)
        
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
        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')

        self.door_bid = self.model.body_name2id('frame')
        self.latch_bid = self.model.body_name2id('latch')
        self.hand_base_sid = self.sim.model.site_name2id('hand_base')
        self.hand_base_bid = self.sim.model.body_name2id('forearm')


        self.ARTz_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARTz')]
        self.ARRx_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARRx')]
        self.ARRy_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARRy')]
        self.ARRz_pos = self.model.jnt_dofadr[self.model.joint_name2id('ARRz')]

    def step(self, a):
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

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def step_statemode(self, desired_state):
        assert desired_state.shape == (28,)
        ctrlrange = self.sim.model.actuator_ctrlrange

        wrist_dof_idx0 = self.sim.model.actuator_name2id('A_ARTz')
        self.sim.data.ctrl[wrist_dof_idx0: wrist_dof_idx0+4] = desired_state[0:4]

        hand_dof_idx0 = self.sim.model.actuator_name2id('pos:A_WRJ1')
        self.sim.data.ctrl[hand_dof_idx0: hand_dof_idx0+24] = desired_state[4:]

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
        return np.concatenate([qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos-handle_pos, [door_open]])

    def get_obs_same_dim(self):
        qp = self.data.qpos.ravel().copy()
        hand_qpos = qp[4:28] # 2 wrist angles+ 22 finger angles
        hand_base_pos = self.data.site_xpos[self.hand_base_sid].ravel().copy()
        hand_base_euler = rotations.quat2euler(self.data.body_xquat[self.hand_base_bid]).ravel().copy()

        handle_pos = self.data.site_xpos[self.handle_sid].ravel().copy()
        handle_euler = rotations.quat2euler(self.data.body_xquat[self.latch_bid]).ravel().copy()

        door_pos = np.array([self.data.qpos[self.door_hinge_did]]) #1-dim
        latch_pos = qp[-1]
        qv = self.data.qvel.ravel().copy()
        latch_vel = qv[-1]

        # return {'hand_qpos':hand_qpos, 'hand_base_t':hand_base_t, 'hand_base_r':hand_base_r, 'obj_pos':obj_pos, 'target_pos':target_pos}
        return np.concatenate([hand_base_pos, hand_base_euler, hand_qpos, handle_pos, handle_euler, door_pos, [latch_pos], [latch_vel]]) # 3+3+24+3+3+1+1+1


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

        obs_dict = {'qp':qp[1:-2], 'arm':np.array([arm_Tz, arm_Rx, arm_Ry, arm_Rz]), 'latch_pos':latch_pos,
                    'door_pos': door_pos, 'palm_pos': palm_pos, 'handle_pos': handle_pos, 'palm_relative_pos':palm_pos-handle_pos,
                    'door_open': door_open}
        return obs_dict

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
        self.sim.forward()
        return self.get_obs()



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
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def render(self, mode='human'):
        self.mj_render()

    def reset(self):
        return self._reset()
