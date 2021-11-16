# import gym
# import mj_envs
# env_name = 'ManipulationPush-v8'
# env = gym.make(env_name)
#
# while True:
#     env.env.mj_render()

# import gym
#
# env_name = 'UR5ePush-v1'
# env = gym.make(env_name)
# env.reset()
#
# for i in range(100):
#     env.render()
#
# while True:
#     env.render()
#     action=[0.0,0.0,0.08,0.0]
#     env.step(action)
#


import threading
from threading import Thread
from threading import Lock
import time
import math
import sys
import pickle
import gym

import copy
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
import numpy as np
import mujoco_py
import we_envs

velocity_scale = 5.0
angularv_scale = 0.01

ee_pos = np.zeros(4, dtype=np.float)

def velocity_callback(velocity):
    global sim, ee_pos, ee_quat
    v = [0, 0, 0, 0, 0, 0]
    v[0] = -velocity.twist.linear.x * velocity_scale
    v[1] = -velocity.twist.linear.y * velocity_scale
    v[2] = velocity.twist.linear.z * velocity_scale
    v[3] = velocity.twist.angular.x * angularv_scale
    v[4] = velocity.twist.angular.y * angularv_scale
    v[5] = velocity.twist.angular.z * angularv_scale

    # ee_pos,ee_quat = cartisian_control.get_ee_state(sim,_SITE_NAME)

    mutex.acquire()
    for i in range(3):
        ee_pos[i] =v[i]

    mutex.release()

    # print(ee_pos)




if __name__ == "__main__":
    rospy.init_node('ur5_mujoco_joy_control')

    global sim, model, mutex, last_force_move, gripper_current_pos, path

    Bmove = False
    mutex = threading.Lock()

    rospy.Subscriber("/velocity", TwistStamped, velocity_callback)

    env_name = 'We_UR5ePushDense-v2'
    env = gym.make(env_name)
    env.reset()

    env_unwrap = env.unwrapped

    while True:
        action = np.zeros(4, dtype=np.float)
        for i in range(3):
            action[i] = ee_pos[i]
        observation, reward, done ,info = env.step(action)
        fdata,tdata=env_unwrap.getFTforce()
        # print(fdata[0], fdata[1], fdata[2])
        # print(reward)


        if info['is_success']:
            print('success')
        # touchforce = env_unwrap.getTouchforce()
        # print(touchforce)
        env.render()
