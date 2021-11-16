import threading
import sys
import time
import pickle
import gym
sys.path.append(".")
import rospy

import numpy as np
import we_envs


import threading
import sys
import time
import pickle
import gym
sys.path.append(".")
import rospy
from geometry_msgs.msg import TwistStamped
from  joints_angle.msg import joints_angle

import math
from we_envs.we_robots.we_utils.rotations import *
import pickle
import os



def set_fixed_goal(env):
	env.set_goal(['s'])
	# pass

getchar=False

def get_c():
    global getchar
    while True:
        iput=input()
        if iput!=None:
            getchar = not getchar



if __name__ == "__main__":


    global  ee_pos
    ee_pos=np.zeros(7,dtype=np.float64)

    global finger_angles
    finger_angles=np.zeros(13,dtype=np.float64)

    t = threading.Thread(target=get_c)
    t.setDaemon(True)
    t.start()

    mutex = threading.Lock()


    env_name = 'We_UR5eShadowLite-v2'
    env = gym.make(env_name)
    # env = env.unwrapped
    env.reset()



    o=env.reset()

    set_fixed_goal(env)

    policy_file = 'demonstration/2019_11_04_15_21_21.pickle'
    data = pickle.load(open(policy_file, 'rb'))
    #print(data)

    obs_buf = data['obs']
    action_buf = data['action']

    #new_dict={}
    #new_dict['actions'] = []
    #new_dict['obs'] = []


    for i in range(action_buf.shape[0]):
        #if i > 100 and i < 350:
            #continue
        env.render()
        print(i)
        #global getchar
        #while  getchar:
            #env.render()

        action = action_buf[i]
        #print(type(action))

        #new_dict['actions'].append(action_buf[i])
        #new_dict['obs'].append(obs_buf[i])


        action = np.array(action, dtype=np.float32)

        obs, reward, done, info = env.step(action)

        # print(obs['observation'][26:30])

    # https://github.com/openai/mujoco-py/issues/10
    # for i in range(1000):
    #     env.render()
    #     env.viewer.cam.lookat[1]+=0.001
    #     env.viewer.cam.elevation+=0.2
    #     env.viewer.cam.azimuth +=0.1


    #print(new_dict)
    #output = open('modified_'+policy_file, 'wb')

    #pickle.dump(new_dict,output)
    #output.close()
    env.close()
