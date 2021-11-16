import threading
import sys
import time
import pickle
import gym
sys.path.append(".")
import rospy
from geometry_msgs.msg import TwistStamped
import numpy as np
import we_envs

velocity_scale = 0.2
angularv_scale = 0.01
min_motiontime = 0.5
average_acc = 0.1


def velocity_callback(velocity):
    global sim, ee_pos, ee_quat
    v = [0, 0, 0, 0, 0, 0]
    v[1] = velocity.twist.linear.x * velocity_scale
    v[0] = -velocity.twist.linear.y * velocity_scale
    v[2] = velocity.twist.linear.z * velocity_scale
    v[3] = velocity.twist.angular.x * angularv_scale
    v[4] = velocity.twist.angular.y * angularv_scale
    v[5] = velocity.twist.angular.z * angularv_scale

    # ee_pos,ee_quat = cartisian_control.get_ee_state(sim,_SITE_NAME)

    mutex.acquire()
    for i in range(6):
        ee_pos[i] =v[i]

    mutex.release()





if __name__ == "__main__":
    rospy.init_node('ur5_mujoco_joy_control')

    global  ee_pos
    ee_pos=np.zeros(6,dtype=np.float64)
    mutex = threading.Lock()

    rospy.Subscriber("/velocity", TwistStamped, velocity_callback)

    env_name = 'We_UR5eShadowLite-v2'
    env = gym.make(env_name)
    env = env.unwrapped
    env.reset()

    goal_done = False


    while True:
        env.reset()
        goal_done = False
        print("desired goal is", env.goal)
        while not goal_done:
            env.render()

            action = np.zeros(19, dtype=np.float)
            for i in range(6):
                action[i] = ee_pos[i]
            # print(action)

            # action = [ee_position_angle, hand_ctrl], is a 19-dim vector
            # hand_ctrl = [FF4, FF3, FF21, MF4, MF3, MF21, RF4, RF3, RF21, TH5, TH4, TH2, TH1]

            # action[9]=-0.5
            action[13] = 1.57


            # for i in range(13):
            #     action[i+6]=0.5
            #     obs, reward, done, info = env.step(action)
            #     # time.sleep(0.8)
            #     for _ in range(40):
            #         env.render()

            obs, reward, done, info = env.step(action)
            # env.display_keyboardData(10)

            # finger_pos_data: [FF4, FF3, FF2, FF1, MF4, MF3, MF2, MF1, RF4, RF3, RF2, RF1, TH5, TH4, TH2, TH1]
            # finger_pressure_data: Ffinger, Mfinger, Rfinger, Thumb
            finger_pos_data, finger_pressure_data = env.getFinger_obs()
            FF3_pos = finger_pos_data[9]
            # print(FF3_pos)
            if done:
                print("done")

            print("desired goal is", env.goal)
            print("achieved goal is", obs['achieved_goal'])
            print("early abort", env.early_abort)
            print("-------------------------")

