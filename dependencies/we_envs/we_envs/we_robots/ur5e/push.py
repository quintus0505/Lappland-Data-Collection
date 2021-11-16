from gym import utils
from we_envs.we_robots.ur5e import ur5e_env2


class WeUR5ePushEnv(ur5e_env2.UR5eEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'joint1': -1.1271217505,
            'joint2': -1.67185988,
            'joint3': -1.6666574478,
            'joint4': -1.37534459,
            'joint5': 1.5675125122,
            'joint6': 3.58500,
        }

        # initial_qpos = {
        #     # 'robot0:slide0': 0.405,
        #     # 'robot0:slide1': 0.48,
        #     # 'robot0:slide2': 0.0,
        #     # 'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        # }
        print('This env is in we_envs')
        
        ur5e_env2.UR5eEnv.__init__(
            self, 'ur5e/ur5e_mocap_new.xml', has_object=True, block_gripper=True, n_substeps=10,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
