import numpy as np
import  time
from we_envs.we_robots.ur5eShadowLite import ur5ehand_base_env
from we_envs.we_robots.we_utils import rotations, utils
from we_envs.we_robots.we_utils.sensor_ur5eshadowlite import  Sensor
import math



class UR5eShadowLiteEnv(ur5ehand_base_env.RobotEnv):
    """Superclass for all UR5eShadowLite environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new UR5eShadowLite environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        
        self.object_site_name = 'egg1'
        # self.gripper_site_name = 'gripper_center'
        self.object_joint_name = 'Jegg1'

        # self.object_site_name = 'object0'
        self.gripper_site_name = 'robot0:grip'
        # self.object_joint_name = 'object0:joint'

        self.gripper_rotation = np.array([0.0, 0.0, 1.0, 0.0])
        self.last_rot_ctrl=rotations.quat2euler(self.gripper_rotation)
        self.last_pos_ctrl=np.zeros(3,dtype=np.float64)
        self._max_episode_steps_2 = 1000
        self.initial_qpos = initial_qpos

        self.goal = []
        self.early_abort = False  # if the keyboard value manipulated is totally wrong, just stop it early!
        self.dangerous = False

        # Configutable parameters
        self.her_style = True #if true, the reset and step function will return more infomation
        self.fixed_goal = True # if true, the key to be typed will always be 'space' every reset


        super(UR5eShadowLiteEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=19,
            initial_qpos=initial_qpos)

        self.initial_force, self.initial_torque = self.getFTforce()
        self.iter = 0
        self.achived_goal = []



    # GoalEnv methods
    # ----------------------------
    def goal_achieved(self, achieved_goal, desired_goal):
        if len(achieved_goal) > len(desired_goal):
            self.early_abort = True
        elif len(achieved_goal) == len(desired_goal):
            # if achieved_goal == desired_goal:
            for i in range(len(achieved_goal)):
                if achieved_goal[i] != desired_goal[i]:
                    self.early_abort = True
                    return False
            return True
        elif len(achieved_goal) < len(desired_goal):
            # if not (achieved_goal == desired_goal[:len(achieved_goal)].all()):
            for i in range(len(achieved_goal)):
                if achieved_goal[i] != desired_goal[i]:
                    self.early_abort=True
        return False


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_achieved(achieved_goal, goal)
        if d:
            reward=0.0
        else:
            reward=-1.0

        # reward = -d.astype(np.float32)
        return   reward


    # RobotEnv methods
    # ----------------------------


    def getFTforce(self):
        """
        :return:
        force_data: np.array(3)
        torque_data: np.array(3)
        """
        force_data,torque_data= self.sensors.get_force_torque_data_tcp()
        return force_data, torque_data

    def get_raw_keyboardData(self):
        keyboard_data = self.sensors.get_keyboard_data()
        keyboard_map = ['esc','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12',
                        'print_screen','scroll_lock','pause_break','insert','home','pageup','delete','end','pagedown',
                        'up','left','down','right','numlock','divide','multiply','subtract','7','8','9','add',
                        '4','5','6','1','2','3','enter_right','0','dot','~',
                        '1!','2@','3#','4$','5%','6^','7and','8*','9(','0)','-_','=+','backspace',
                        'tab','q','w','e','r','t','y','u','i','o','p','[',']','vertical',
                        'caps_lock','a','s','d','f','g','h','j','k','l',';','double_quotation','enter_left',
                        'shift_left','z','x','c','v','b','n','m','comma','point','question_mark','shift_right',
                        'ctrl_left','win_left','alt_left','space','alt_right','win_right','application','ctrl_right']
        keyboard_data_dict=dict(zip(keyboard_map,keyboard_data))
        return keyboard_data_dict

    def resolve_raw_keyboardData(self, keyboard_data_dict):
        for key, value in keyboard_data_dict.items():
            keyboard_data_dict[key] = 'pressed' if value > 0.007 else 'released'
        return keyboard_data_dict

    def get_pressed_keyboardData(self, keyboard_data_dict):
        keyboard_data_list = []
        Upper_flag = False if keyboard_data_dict['ctrl_left'] is 'released' else True
        letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                   'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                   'z', 'x', 'c', 'v', 'b', 'n', 'm', 'space']
        for key, value in keyboard_data_dict.items():
            if value is 'pressed':
                if key in letters and Upper_flag:
                    keyboard_data_list.append(key.upper())
                else:
                    keyboard_data_list.append(key)
        if Upper_flag:
            keyboard_data_list.remove("ctrl_left")
        return keyboard_data_list

    def get_final_keyboardData(self, T=10, counter=[1]):
        keyboard_raw_data = self.get_raw_keyboardData()
        keyboard_resolved_data = self.resolve_raw_keyboardData(keyboard_raw_data)
        keyboard_info = self.get_pressed_keyboardData(keyboard_resolved_data)
        # need optimize ,set should be added on each step or it just detect the last step
        counter[0] = 1 if counter[0] == T else counter[0] + 1
        if counter[0] == 1:
            return keyboard_info

    def display_keyboardData(self, T=10, counter=[1]):
        keyboard_raw_data = self.get_raw_keyboardData()
        keyboard_resolved_data = self.resolve_raw_keyboardData(keyboard_raw_data)
        keyboard_info = self.get_pressed_keyboardData(keyboard_resolved_data)
        # need optimize ,set should be added on each step or it just detect the last step
        counter[0] = 1 if counter[0] == T else counter[0] + 1
        if counter[0] == 1:
            print("The pressed keys are ", end='')
            print(keyboard_info)


    def getFTforce_tcp(self):
        force_data, torque_data = self.sensors.get_force_torque_data_tcp()
        return force_data, torque_data

    def safety_constraint(self):
        # return false if torque or force over the limit
        force_data , torque_data = self.getFTforce_tcp()
        # print(force_data)
        # print(torque_data)
        # print("-----------")
        dangerous = np.max(np.fabs(force_data))>900.0 and np.max(np.fabs(torque_data))>200.0
        return dangerous
    
    def getFinger_obs(self):
        """
        :return:
        finger_pos_data: [FF4, FF3, FF2, FF1, MF4, MF3, MF2, MF1, RF4, RF3, RF2, RF1, TH5, TH4, TH2, TH1]
        finger_pressure_data: Ffinger, Mfinger, Rfinger, Thumb
        """
        finger_pos_data, finger_pressure_data=self.sensors.get_finger_data()
        return  finger_pos_data, finger_pressure_data


    def _step_callback(self):
        if self.block_gripper:
            # self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            # self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (19,)
        action_temp = action.copy()  # ensure that we don't change the action outside of this scope

        pos_inc_ctrl, rot_inc_ctrl = action_temp[:3], action_temp[3:6]

        # look for hand_lite_technical_specification,
        #  hand_ctrl=[FF4,FF3,FF21, MF4, MF3,MF21, RF4, RF3, RF21, TH5, TH4, TH2, TH1]
        hand_ctrl = action_temp[6:19]

        # pos_inc_ctrl *= 0.05  # limit maximum change in position, is the increamental value
        self.last_rot_ctrl +=rot_inc_ctrl
        quat_abs_ctl = rotations.euler2quat(self.last_rot_ctrl)
        self.last_pos_ctrl +=pos_inc_ctrl

        action_ur5e = np.concatenate([self.last_pos_ctrl.copy(), quat_abs_ctl])

        # Apply action to simulation.
        utils.mocap_set_action(self.sim, action_ur5e)

        utils.ctrl_set_action(self.sim, hand_ctrl)

    def mcs_demonstration(self, pos_inc, quat, hand_ctrl):

        self.last_pos_ctrl+=pos_inc
        self.last_rot_ctrl = rotations.quat2euler(quat)

        action_ur5e = np.concatenate([self.last_pos_ctrl.copy(), quat])
        utils.mocap_set_action(self.sim, action_ur5e)
        utils.ctrl_set_action(self.sim, hand_ctrl)
        return  self.mcs_step()




    def get_gripper_velocity(self):
        gripper_velp = self.sim.data.get_site_xvelp(self.gripper_site_name)
        return gripper_velp

    def get_object_velocity(self):
        object_velp = self.sim.data.get_site_xvelp(self.object_site_name)
        return object_velp

    def get_hand_state(self):
        finger_pos_data, finger_pressure_data=self.sensors.get_finger_data()
        return finger_pos_data, finger_pressure_data

    def _get_obs(self):
        # positions
        hand_pos = self.sim.data.get_site_xpos(self.gripper_site_name)
        hand_quat =  rotations.mat2quat(self.sim.data.get_site_xmat(self.gripper_site_name)) #TODO: is equal to input action? need test!
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        hand_velp = self.sim.data.get_site_xvelp(self.gripper_site_name) * dt
        finger_pos_data, finger_pressure_data = self.get_hand_state()

        keyboard_state = self.get_final_keyboardData()

        obs = np.concatenate([
            hand_pos, hand_quat, hand_velp, finger_pos_data, finger_pressure_data
            # ,keyboard_state
        ])

        if not keyboard_state is None:
            self.achived_goal+=keyboard_state
            # print(keyboard_state)

        self.dangerous = self.safety_constraint()

        return {
            'achieved_goal':  self.achived_goal,
            'observation': obs.copy(),
            'desired_goal': self.goal.copy(),
        }



    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:r_gripper_finger_link')
        body_id = self.sim.model.body_name2id('ee_link')

        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.

        self.sim.forward()

    def _reset_sim(self):
        self.gripper_rotation = np.array([0.0, 0.0, 1.0, 0.0])
        self.last_rot_ctrl = rotations.quat2euler(self.gripper_rotation)
        self.last_pos_ctrl = np.zeros(3, dtype=np.float64)
        self.goal = []
        self.initial_force, self.initial_torque = self.getFTforce()
        self.iter = 0
        self.achived_goal = []
        self.early_abort = False
        self.dangerous = False

        self._env_setup(self.initial_qpos)

        self.sim.set_state(self.initial_state)
        self.goal=self._sample_goal()

        self.sim.forward()
        return True

    def set_goal(self, letters_input):
        if self.fixed_goal:
            self.goal = np.array(['space'])
        else:
            letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                       'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                       'z', 'x', 'c', 'v', 'b', 'n', 'm']
            letters_upper = [key.upper() for key in letters]
            letters_all = letters + letters_upper+['space']
            for key in letters_input:
                assert key in letters_all, "letter inputs are not char!!"
            # self.reset()
            self.goal = np.array(letters_input)


    def _sample_goal(self):
        if self.fixed_goal:
            self.set_goal(['space'])
            return self.goal
        else:
            letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                       'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                       'z', 'x', 'c', 'v', 'b', 'n', 'm']
            letters_upper = [key.upper() for key in letters]
            letters_all = letters+letters_upper

            printstring_len = np.random.randint(low=1, high=11) #TODO: high means the max count need to be typed by dexterous hand
            assert printstring_len > 0
            printstring = np.random.choice(letters_all, printstring_len)
            goal = printstring
            return goal.copy()


    def _is_success(self, achieved_goal, desired_goal):
        return self.goal_achieved(achieved_goal, desired_goal)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        gripper_target = self.sim.data.get_site_xmat(self.gripper_site_name)

        # Move end effector into position.
        gripper_target = np.array([0.0, 0.0, 0.0 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(self.gripper_site_name)
        gripper_rotation = self.gripper_rotation


        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        self.last_pos_ctrl = gripper_target
        # self.goal = []
        self.achived_goal = []
        self.early_abort = False
        self.dangerous = False
        
        for _ in range(30):
            self.sim.step()


        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos(self.gripper_site_name).copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos(self.object_site_name)[2]
