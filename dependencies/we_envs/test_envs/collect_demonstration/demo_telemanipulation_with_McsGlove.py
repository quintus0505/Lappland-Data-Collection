import threading
import sys
import time
import pickle
import gym
sys.path.append(".")
import rospy
from geometry_msgs.msg import TwistStamped
from  joints_angle.msg import joints_angle
from mcs_driver.msg import RigidBody
import numpy as np
import we_envs
import math
from we_envs.we_robots.we_utils.rotations import *
import pickle
import os

velocity_scale = 0.8
min_motiontime = 0.5
average_acc = 0.1

base_position = np.zeros(3, dtype=np.float)
base_euler = np.zeros(3, dtype=np.float)
inited = False


def mcs_callback(RigidBodyPose):
    global sim, ee_pos, ee_quat, base_position, base_euler, inited


    if not (RigidBodyPose.name =="rigid_body_2"):
        return



    abs_pos = np.zeros(3, dtype=np.float)
    abs_pos[0] = RigidBodyPose.pose.pose.position.x
    abs_pos[1] = RigidBodyPose.pose.pose.position.y
    abs_pos[2] = RigidBodyPose.pose.pose.position.z

    abs_quat = np.zeros(4, dtype=np.float)
    abs_quat[0] = RigidBodyPose.pose.pose.orientation.w
    abs_quat[1] = RigidBodyPose.pose.pose.orientation.x
    abs_quat[2] = RigidBodyPose.pose.pose.orientation.y
    abs_quat[3] = RigidBodyPose.pose.pose.orientation.z

    abs_euler = quat2euler(abs_quat)

    if not inited:
        base_position[:] = abs_pos[:]
        base_euler[:] = abs_euler
        inited = True
        return

    inc_pos = abs_pos - base_position
    inc_euler = abs_euler - base_euler

    # for i in range(3):
    #     if abs(inc_euler[i])>=100:
    #         inc_euler[i] = 0

    v = np.zeros(6, dtype=np.float)
    v[0] = inc_pos[0] * velocity_scale
    v[1] = inc_pos[1] * velocity_scale
    v[2] = inc_pos[2] * velocity_scale
    v[3] = inc_euler[0]
    v[4] = inc_euler[1]
    v[5] = inc_euler[2]

    base_position = abs_pos[:]
    base_euler = abs_euler[:]

    mutex.acquire()
    for i in range(6):
        ee_pos[i] =v[i]
    mutex.release()

def shadowhand_tele_callback(finger_angles_input):
    global finger_angles
    mutex.acquire()
    # TODO:
    finger_angles[12] = finger_angles_input.T1  # T1
    finger_angles[11] = -(finger_angles_input.T2 + 25.0) * 1.5  # T2
    finger_angles[10] = -(finger_angles_input.T3 + finger_angles_input.T4 - 5)  # T4
    finger_angles[9] = -(finger_angles_input.T5 - 55.0)  # T5

    finger_angles[2] = 0.25 * finger_angles_input.F1 + 0.75 * finger_angles_input.F2 - 3  # F1
    finger_angles[1] = finger_angles_input.F3  # F2
    finger_angles[0] = -finger_angles_input.F4  # F3

    finger_angles[5] = 0.25 * finger_angles_input.M1 + 0.75 * finger_angles_input.M2 - 3  # M1
    finger_angles[4] = finger_angles_input.M3  # M2
    finger_angles[3] = -finger_angles_input.M4  # M3

    finger_angles[8] = 0.4 * finger_angles_input.R1 + 0.55 * finger_angles_input.R2 - 3  # R1
    finger_angles[7] = finger_angles_input.R3  # R2
    finger_angles[6] = -finger_angles_input.R4  # R3

    mutex.release()

def set_fixed_goal(env):
	env.set_goal(['space'])
	# pass


if __name__ == "__main__":
    rospy.init_node('ur5e_mujoco_telemanipulation_control')

    global  ee_pos
    ee_pos=np.zeros(7,dtype=np.float64)

    global finger_angles
    finger_angles=np.zeros(13,dtype=np.float64)


    mutex = threading.Lock()

    rospy.Subscriber("/mcs_pose", RigidBody, mcs_callback)
    rospy.Subscriber("/RealJointsAngle", joints_angle, shadowhand_tele_callback)

    env_name = 'We_UR5eShadowLite-v2'
    env = gym.make(env_name)
    # env = env.unwrapped
    env.reset()

    goal_done = False

    o=env.reset()
    o = o['observation']
    set_fixed_goal(env)
    goal_done = False
    print("desired goal is", env.goal)

    obs_buf = []
    action_buf = []

    # when the goal is achived, it not mean that the type is correct, we should delay to observe whether the finger is going up
    done_iter=0
    while not goal_done:
        env.render()

        action = np.zeros(19, dtype=np.float)
        for i in range(6):
            action[i] = ee_pos[i]

        # action = [ee_position_angle, hand_ctrl], is a 19-dim vector
        # hand_ctrl = [FF4, FF3, FF21, MF4, MF3, MF21, RF4, RF3, RF21, TH5, TH4, TH2, TH1]
        for j in range(13):
            action[6 + j] = finger_angles[j] * math.pi / 180.0


        obs_buf.append(o)
        action_buf.append(action)

        obs, reward, done, info = env.step(action)
        o = obs['observation']
        print(env.achived_goal)

        if done:
            done_iter=done_iter+1

        if done_iter>30:
            goal_done =True

    if goal_done:
        obs_buf = np.array(obs_buf, dtype=np.float32)
        action_buf = np.array(action_buf, dtype=np.float32)

        sa_pair = {"obs": obs_buf, "action": action_buf}

        file_path = './demonstration/' \
                    + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())) + '.pickle'
        with open(file_path, 'wb') as f:
            pickle.dump(sa_pair, f)

    env.close()

