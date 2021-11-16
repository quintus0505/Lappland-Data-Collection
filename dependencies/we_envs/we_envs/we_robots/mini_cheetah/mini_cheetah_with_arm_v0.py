import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
from we_envs.we_robots.we_utils.rotations import *

ADD_BONUS_REWARDS = True
# if use mujoco-quat, the unit quaternion is [1,0,0,0], if use Mit/pybullet definition is [0,0,0,1]
USE_MUJOCO_QUATERNION = False

def quat_mujoco2bullet(mujoco_quat):
    bullet_quat = np.array([mujoco_quat[1], mujoco_quat[2], mujoco_quat[3],mujoco_quat[0]])
    return bullet_quat

def quat_bullet2mujoco(bullet_quat):
    mujoco_quat = np.array([bullet_quat[3], bullet_quat[0], bullet_quat[1],bullet_quat[2]])
    return mujoco_quat

class MiniCheetahWithArmEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/../assets/mini_cheetah/with_arms/mini_cheetah_withArm.xml', 2)
        
        # change actuator sensitivity
        # self.sim.model.actuator_gainprm[
        # self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0') + 1, 0:3] = np.array(
        #     [10, 0, 0])
        # self.base_bid = self.sim.model.body_name2id("base")
        # self.imu_sid = self.sim.model.site_name2id('imu')

        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

        self.init_state = dict()

    def step(self, a, NOT_USR_ARM=True, arm_ctrl=[0.0]*9):
        if NOT_USR_ARM:
            if len(a)==12: # trigged when init function is called
                arm_ctrl = np.array([0.0]*9, dtype=np.float)
                a = np.concatenate([a,arm_ctrl])
        else:
            arm_ctrl = np.array(arm_ctrl, dtype=np.float)
            a = np.concatenate([a,arm_ctrl])
        a = np.clip(a, self.model.actuator_ctrlrange[:,0], self.model.actuator_ctrlrange[:,1])
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()

        reward = 0
        goal_achieved = False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_joint_torque(self):
        # self.fr_jid = self.sim.model.joint_name2id("torso_to_abduct_fr_j")
        return self.data.qfrc_unc[6:]+self.data.qfrc_constraint[6:]

    def get_leg_state(self):
        qp = self.data.qpos.ravel()
        qv = self.data.qvel.ravel()
        leg_data = np.concatenate([qp[7:19], qv[6:18]])
        return leg_data

    def get_base_pos(self):
        qp = self.data.qpos.ravel().copy()
        if not USE_MUJOCO_QUATERNION:
            base_quat = quat_mujoco2bullet(qp[3:7])
        else:
            base_quat = qp[3:7]
        base_pos = {'loc':qp[0:3], 'orn':base_quat}
        return base_pos

    def get_base_vel(self):
        qv = self.data.qvel.ravel().copy()
        base_vel = qv[0:3]
        return base_vel

    def get_imu_data(self):
        """
        :return:
            linear acceleration [3], quaternion [4],  angular velocity [3] in local coordinates
            (pybullet read the base's data in global coordinates and tranform to local coordinates ,done by user self)
        """
        sensordata = self.data.sensordata.copy()
        imu_quaternion = sensordata.ravel()[0:4]
        if not USE_MUJOCO_QUATERNION:
            imu_quaternion = quat_mujoco2bullet(imu_quaternion)
        angular_velocity = sensordata.ravel()[4:7]
        linear_acceleration = sensordata.ravel()[7:10]
        return np.concatenate([linear_acceleration, imu_quaternion, angular_velocity])


    def get_obs(self):
        base_loc = self.data.qpos.ravel()[0:3]
        base_quat = self.data.qvel.ravel()[3:7].copy()
        if not USE_MUJOCO_QUATERNION:
            base_quat = quat_mujoco2bullet(base_quat)
        imu_data = self.get_imu_data()
        leg_data = self.get_leg_state()
        return np.concatenate([base_loc, base_quat, imu_data,leg_data])

    def set_init_state(self, joint_state, base_state=[0,0,0,1, 0,0,0,0]):
        assert len(joint_state)==24 and len(base_state)==7, 'input joint pos dim is not correct!'
        joint_qp = joint_state[0:12]
        joint_qv = joint_state[12:24]
        base_qp = base_state[0:4]
        if not USE_MUJOCO_QUATERNION:
            base_qp = quat_bullet2mujoco(base_qp)
        base_qv = base_state[3:6]
        self.init_qpos=np.concatenate([base_qp, joint_qp])
        self.init_qvel=np.concatenate([base_qv, joint_qv])

    def set_base_state(self, base_loc, base_quat, base_vel=[0,0,0], base_angular_vel=[0,0,0]):
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        base_quat_mujoco = base_quat
        if not USE_MUJOCO_QUATERNION:
            base_quat_mujoco = quat_bullet2mujoco(base_quat)
        qp[0:7]=np.concatenate([base_loc, base_quat_mujoco])
        qv[0:6]=np.concatenate([base_vel, base_angular_vel])
        self.set_state(qp, qv)
        # self.model.body_pos[self.obj_bid] = obj_pos
        for _ in range(2):
            self.sim.forward()
            self.render()

    def set_joint_state(self, joint_state):
        joint_qp = joint_state[0:12]
        joint_qv = joint_state[12:24]
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        qp[7:19]=joint_qp
        qv[6:18]=joint_qv
        self.set_state(qp, qv)

        for _ in range(2):
            self.sim.forward()
            self.render()

       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        # self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15*self.scale, high=0.15*self.scale)

        for _ in range(1):
            self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()

        return self.get_obs()


    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()

        base_loc = self.data.qpos.ravel()[0:3]
        base_quat = self.data.qvel.ravel()[3:7].copy()
        if not USE_MUJOCO_QUATERNION:
            base_quat = quat_mujoco2bullet(base_quat)

        joint_pos = qp[7:]
        joint_vel = qv[6:]
        return dict(qpos=qp, qvel=qv, base_pos=base_loc, base_quat= base_quat, joint_po=joint_pos, joint_vel=joint_vel)

    # def debug(self):
    #     state_dict = self.get_env_state().copy()
    #     base_pos = state_dict['base_pos']
    #     base_pos_2 = self.data.qpos.ravel().copy()[0:3]
    #     print(base_pos-base_pos_2)
    #     return

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qp']
        if not USE_MUJOCO_QUATERNION:
            qp[3:7] = quat_bullet2mujoco(qp[3:7])
        qv = state_dict['qv']
        self.set_state(qp, qv)
        # self.model.body_pos[self.obj_bid] = obj_pos
        self.sim.forward()
        # for _ in range(500):
        #     self.sim.step()


    def mj_viewer_setup(self, tracking=True):
        self.viewer = MjViewer(self.sim)
        if tracking:
            from mujoco_py.generated import const
            self.viewer.cam.type = const.CAMERA_TRACKING
            self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.distance = self.model.stat.extent * 1.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
        # self.viewer.cam.lookat[0] -= 5         # x,y,z offset from the object (works if trackbodyid=-1)
        # self.viewer.cam.lookat[1] += 0.5
        # self.viewer.cam.lookat[2] += 0.5
        # self.viewer.cam.elevation = -20          # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        # self.viewer.cam.azimuth = 0              # camera rotation around the camera's vertical axis
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5


    def render(self, mode='human'):
        self.mj_render()

    def reset(self):
        return self._reset()
