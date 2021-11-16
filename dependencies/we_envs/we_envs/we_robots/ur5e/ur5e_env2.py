import numpy as np
import  time
from we_envs.we_robots.we_utils import rotations
from we_envs.we_robots.ur5e import robot_env, utils_ur5e_push
from we_envs.we_robots.we_utils.sensor import  Sensor
import math


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5eEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.

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
        
        self.object_site_name = 'cube_1'
        # self.gripper_site_name = 'gripper_center'
        self.object_joint_name = 'Jcube1'

        # self.object_site_name = 'object0'
        self.gripper_site_name = 'robot0:grip'
        # self.object_joint_name = 'object0:joint'

        self.gripper_rotation = np.array([0.0, 1.0, 0.0, -1.0])

        # Configutable parameters
        self.fix_env = True
        self.reward_type_Dense = True
        self.her_style = False  #if true, the reset and step function will return more infomation


        self.initial_qpos=initial_qpos

        super(UR5eEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=3,
            initial_qpos=initial_qpos)
        self.sensors = Sensor(self.sim)
        self.initial_force, self.initial_torque = self.getFTforce()
        self.iter = 0

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        present_force, present_torque = self.getFTforce()
        delta_force = present_force-self.initial_force
        #if (np.linalg.norm(delta_force) < 80):
            #print('hehe',end='')
            #self.iter = np.linalg.norm(delta_force)
            #print(self.iter)

        if self.reward_type == 'sparse' and not self.reward_type_Dense:
            touchforce = self.getTouchforce()
            touchforce_have = touchforce[0]>0.1 or touchforce[1]>0.1
            if touchforce_have:
                touchforce_gain = 1.0
            else:
                touchforce_gain = 0.0

            reward = -(d > self.distance_threshold).astype(np.float32)
            reward += touchforce_gain*0.1
            reward -= (np.linalg.norm(delta_force) > 80).astype(np.float32)*0.1
            return   reward

        else:
            touchforce = self.getTouchforce()
            touchforce_have = touchforce[0] > 0.1 or touchforce[1] > 0.1
            if touchforce_have:
                touchforce_gain = 1.0
            else:
                touchforce_gain = 0.0
            obs = self._get_obs()
            obs = obs['observation']
            distance_grip2obj = np.linalg.norm(obs[6:9], axis=-1)
            reward_distance_grip2obj =  -distance_grip2obj if -distance_grip2obj<-0.10 else 0.0
            reward_distance_grip2obj*=2.0

            reward = -d*5.0
            reward += reward_distance_grip2obj
            # reward += touchforce_gain * 0.01
            # reward -= (np.linalg.norm(delta_force) > 90).astype(np.float32) * 0.05
            return reward

    # RobotEnv methods
    # ----------------------------
    def test_unwrap(self):
        print('hehe')

    def getFTforce(self):
        """
        :return:
        force_data: np.array(3)
        torque_data: np.array(3)
        """
        force_data,torque_data= self.sensors.get_force_torque_data()
        return force_data, torque_data

    def getFTforce_tcp(self):
        force_data, torque_data = self.sensors.get_force_torque_data_tcp()
        return force_data, torque_data
    
    def getTouchforce(self):
        touchforce = self.sensors.get_touchforce()  # np.array(2), left and right

        touchforce_combine = np.zeros(2, dtype=np.float)
        touchforce_combine[0] = max(touchforce[0], touchforce[1])
        touchforce_combine[1] = max(touchforce[2], touchforce[3])
        return touchforce_combine


    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (3,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = self.gripper_rotation  # fixed rotation of the end effector, expressed as a quaternion

        action = np.concatenate([pos_ctrl, rot_ctrl])

        # Apply action to simulation.
        utils_ur5e_push.ctrl_set_action(self.sim, action)
        utils_ur5e_push.mocap_set_action(self.sim, action)

    def get_gripper_velocity(self):
        gripper_velp = self.sim.data.get_site_xvelp(self.gripper_site_name)
        return gripper_velp

    def get_object_velocity(self):
        object_velp = self.sim.data.get_site_xvelp(self.object_site_name)
        return object_velp

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos(self.gripper_site_name)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp(self.gripper_site_name) * dt
        robot_qpos, robot_qvel = utils_ur5e_push.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos( self.object_site_name)
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_site_name))
            # velocities
            object_velp = self.sim.data.get_site_xvelp(self.object_site_name) * dt
            object_velr = self.sim.data.get_site_xvelr(self.object_site_name) * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        # obs = np.concatenate([
        #     grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        #     object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        # ])

        goal_distance = self.goal-object_pos

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(),  object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp,
            # goal_distance.ravel()
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:r_gripper_finger_link')
        body_id = self.sim.model.body_name2id('robot0:r_gripper_finger_link')

        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        if not self.fix_env:
            # Randomize start position of object.
            if self.has_object:
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                object_qpos = self.sim.data.get_joint_qpos(self.object_joint_name)
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos

                # object_qpos[1]=np.clip(object_qpos[1], 0.5, 0.68)
                # euler_angles = rotations.quat2euler(object_qpos[3:7])
                # euler_angles[2]=self.np_random.uniform(-3.14, 3.14,size=1)
                # quat=rotations.euler2quat(euler_angles)
                # object_qpos[3:7]=quat[:4]

                self.sim.data.set_joint_qpos(self.object_joint_name, object_qpos)

        else:
            if self.has_object:
                object_xpos = self.initial_gripper_xpos[:2]+[0.05, 0.05]
                object_qpos = self.sim.data.get_joint_qpos(self.object_joint_name)
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                object_qpos[1] = np.clip(object_qpos[1], 0.5, 0.68)

                euler_angles = rotations.quat2euler(object_qpos[3:7])
                quat = rotations.euler2quat(euler_angles)
                object_qpos[3:7] = quat[:4]
                self.sim.data.set_joint_qpos(self.object_joint_name, object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.fix_env:
            goal = self.initial_gripper_xpos[:3] + np.array([-0.05, 0, 0],dtype=np.float32)
            goal[2] = self.height_offset

            # TODO: only push forward, hehe
            object_qpos = self.sim.data.get_joint_qpos(self.object_joint_name)
            goal[1] = object_qpos[1] +  0.12
            goal = np.array(goal, dtype=np.float32)
        else:
            if self.has_object:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

                #TODO: only push forward, hehe
                object_qpos = self.sim.data.get_joint_qpos(self.object_joint_name)
                goal[1] =object_qpos[1]+ self.np_random.uniform(0, 0.25, size=1)+0.02
            else:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils_ur5e_push.reset_mocap_welds(self.sim)
        self.sim.forward()

        gripper_target = self.sim.data.get_site_xmat(self.gripper_site_name)

        # Move end effector into position.
        gripper_target = np.array([0.0, 0.0, 0.0 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(self.gripper_site_name)
        gripper_rotation = self.gripper_rotation

        # euler2 = np.array([3.14, 0, 0], dtype=np.float)
        # quat2 = rotations.euler2quat(euler2)
        # gripper_rotation = rotations.quat_mul(gripper_rotation, quat2)

        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        for _ in range(30):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos(self.gripper_site_name).copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos(self.object_site_name)[2]
