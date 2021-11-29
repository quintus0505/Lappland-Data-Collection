import numpy as np
from gym import utils
from we_envs.we_robots.AdroitHand.utils import mujoco_env
from mujoco_py import MjViewer
import os
from we_envs.we_robots.we_utils import rotations

ADD_BONUS_REWARDS = True

class DoorEnvV6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0

        self.use_state_same_dim = False
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

    def set_primitive_name(self, primitive_name):
        self.primitive_name = primitive_name

    def set_state_same_dim(self, use_state_same_dim):
        self.use_state_same_dim = use_state_same_dim
        # when the env-observation-dim is changed, following function must be called to get the new correct dim.
        self.get_env_dim()

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
        return True

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
        # reward -= 0.1
        reward_tb = np.zeros(5, dtype=np.float)
        assert not self.goal is None, "please set the goal-of-primitive for envirnment first"
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


    def step_statemode(self, desired_state):
        assert desired_state.shape == (28,)
        ctrlrange = self.sim.model.actuator_ctrlrange

        wrist_dof_idx0 = self.sim.model.actuator_name2id('A_ARTz')
        self.sim.data.ctrl[wrist_dof_idx0: wrist_dof_idx0+4] = desired_state[0:4]

        hand_dof_idx0 = self.sim.model.actuator_name2id('pos:A_WRJ1')
        self.sim.data.ctrl[hand_dof_idx0: hand_dof_idx0+24] = desired_state[4:]

        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        self.sim.step()
        self.sim.step()

    def get_obs(self):
        if self.use_state_same_dim:
            return self.get_obs_same_dim()
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

    def set_noise_scale(self, scale):
        self.scale = scale

    def reset_primtive_env(self, begin_state):
        qp = begin_state['qpos']
        qv = begin_state['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = begin_state['door_body_pos']
        self.data.site_xpos[self.handle_sid] = begin_state['handle_pos']
        self.data.site_xpos[self.grasp_sid] = begin_state['palm_pos']
        self.data.qpos[self.door_hinge_did] = begin_state['door_hinge_pos']
        self.data.qpos[-1] = begin_state['latch_pos']
        for _ in range(500):
            self.sim.forward()
        self.init_state = self.get_env_state()
        return self.get_obs()

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        def set_scale_bound(original_low, original_high, scale):
            assert scale>0, "noise scale is not correct!"
            if scale<0.1:
                scale = 0.1
                print('At present noise scale should not below 0.1, maybe set noise range too small?')
            scaled_range = (original_high- original_low)*(scale-1)/2
            new_low = original_low-scaled_range
            new_high = original_high+scaled_range
            return [new_low, new_high]

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=set_scale_bound(-0.3, -0.2, self.scale)[0],
                                                                      high=set_scale_bound(-0.3, -0.2, self.scale)[1])
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=set_scale_bound(0.25, 0.35, self.scale)[0],
                                                                      high=set_scale_bound(0.25, 0.35, self.scale)[1])

        z_scale_bound = set_scale_bound(0.253, 0.35, self.scale)
        if z_scale_bound[0]<0.253:
            z_scale_bound[0]=0.253
        if z_scale_bound[1]<=z_scale_bound[0]:
            z_scale_bound[1]=z_scale_bound[0]+0.05
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=z_scale_bound[0],
                                                                      high=z_scale_bound[1])

        # self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3*self.scale, high=-0.2*self.scale)
        # self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25*self.scale, high=0.35*self.scale)
        # self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.253, high=0.35)
        self.sim.forward()
        self.init_state = self.get_env_state()
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitive_name=''
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
        latch_pos = self.data.qpos[-1] # The latch joint angle

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

    def render(self, mode='human', extra_info=None):
        self.mj_render()
        self.render_text(name='Primitive', text=self.primitive_name, location='top_left')
        if extra_info is not None:
            for key in extra_info.keys():
                value = extra_info[key]
                self.render_text(name=key, text=str(value), location='top_left', interval=0)

    def reset(self):
        return self._reset()

    def reset_model_initial_state_randomness(self, state_randomness):
        """
        reset env with different noise area for training
        :param state_randomness: noise level area, 1~2.5
        :return: observation of reset env
        """
        assert state_randomness >= 1, "initial state randomness must >=1"
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        def set_scale_bound(original_low, original_high, scale):
            assert scale>0, "noise scale is not correct!"
            if scale<0.1:
                scale = 0.1
                print('At present noise scale should not below 0.1, maybe set noise range too small?')
            scaled_range = (original_high- original_low)*(scale-1)/2
            new_low = original_low-scaled_range
            new_high = original_high+scaled_range
            return [new_low, new_high]

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=set_scale_bound(-0.25-0.03*state_randomness, -0.25+0.03*state_randomness, self.scale)[0],
                                                                      high=set_scale_bound(-0.25-0.03*state_randomness,-0.25+0.03*state_randomness, self.scale)[1])
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=set_scale_bound(0.3-0.035*state_randomness, 0.3+0.035*state_randomness, self.scale)[0],
                                                                      high=set_scale_bound(0.3-0.035*state_randomness, 0.3+0.035*state_randomness, self.scale)[1])

        z_scale_bound = set_scale_bound(0.253, 0.35, self.scale)
        if z_scale_bound[0]<0.253:
            z_scale_bound[0]=0.253
        if z_scale_bound[1]<=z_scale_bound[0]:
            z_scale_bound[1]=z_scale_bound[0]+0.05
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=z_scale_bound[0],
                                                                      high=z_scale_bound[1])
        self.sim.forward()
        self.init_state = self.get_env_state()
        self.primitives_goal_achieved = [0, 0, 0]
        self.primitive_name=''
        return self.get_obs()