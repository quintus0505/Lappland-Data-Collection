import time
import math

from mujoco_py import load_model_from_xml, MjSim, functions, MjViewer
import mujoco_py
import numpy as np
# import gym.envs.we_robots.new_kinematics_ur5e as kinematics
import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from we_envs.we_robots.we_utils.new_kinematics_ur5e import *
import we_envs.we_robots.we_utils.rotations

class Sensor(object):
    def __init__(self,sim):
        self.sim=sim
        self.sensorsdata = None
        self.force_data_base=np.zeros(3, dtype=np.float)
        self.torque_data_base = np.zeros(3,dtype=np.float)

    def getSensorsData(self):
        self.sensors_data = self.sim.data.sensordata


    def get_force_torque_data(self):
        force_data, torque_data = self.get_force_torque_data_tcp()

        sim = self.sim
        base_joint = sim.model.get_joint_qpos_addr("joint1")
        shoulder_joint_i = sim.model.get_joint_qpos_addr("joint2")
        foerarm_joint_i = sim.model.get_joint_qpos_addr("joint3")
        wrist_joint_x = sim.model.get_joint_qpos_addr("joint4")
        wrist_joint_y = sim.model.get_joint_qpos_addr("joint5")
        wrist_joint_z = sim.model.get_joint_qpos_addr("joint6")

        sim_state = sim.get_state()


        th = np.matrix([[sim_state.qpos[base_joint]],
                        [sim_state.qpos[shoulder_joint_i]],
                        [sim_state.qpos[foerarm_joint_i]],
                        [sim_state.qpos[wrist_joint_x]],
                        [sim_state.qpos[wrist_joint_y]],
                        [sim_state.qpos[wrist_joint_z]]])
        c=[0]

        T_06 = HTrans(th,c)
        T_06 = URbase2rosbase(T_06)

        # rotationMat = np.array([[T_06[0,0],T_06[0,1],T_06[0,2]],
        #                         [T_06[1,0],T_06[1,1],T_06[1,2]],
        #                         [T_06[2,0],T_06[2,1],T_06[2,2]]])
        # euler = rotations.mat2euler(rotationMat)
        # print(euler)
        #
        # print(T_06[0,3], T_06[1,3], T_06[2,3])

        force_data_trans = np.matrix([[force_data[0]], [force_data[1]], [force_data[2]], [1]])
        torque_data_trans = np.matrix([[torque_data[0]], [torque_data[1]], [torque_data[2]], [1]])

        try:
            force_data_base = T_06 * force_data_trans
            torque_data_base = T_06 * torque_data_trans
            self.force_data_base = force_data_base
            self.torque_data_base = torque_data_base
        except:
            return  self.force_data_base, self.torque_data_base

        return force_data_base, torque_data_base



    def get_force_torque_data_tcp(self):
        self.getSensorsData()
        # print(self.sensors_data)
        force_torque_data=self.sensors_data[136:142]
        force_data = np.zeros(3,dtype=np.float32)
        torque_data = np.zeros(3,dtype=np.float)
        for i in range(3):
            force_data[i]=-force_torque_data[i]
            torque_data[i]=-force_torque_data[i+3]
        return force_data,torque_data
    
    def get_Touchforce(self):
        self.getSensorsData()
        touchdata = np.zeros(4,dtype=np.float)
        touchdata[0]=self.sensors_data[6] #forward
        touchdata[1]=self.sensors_data[7] #backward
        touchdata[2]=self.sensors_data[8] #left
        touchdata[3] = self.sensors_data[9]  #right
        return touchdata

    def get_touchforce(self):
        return self.get_Touchforce()
        
    def get_keyboard_data(self):
        self.getSensorsData()
        keyboard_data = np.zeros(104,dtype=np.float64)
        keyboard_data=self.sensors_data[32:136]
        return keyboard_data

    def get_finger_data(self):
        self.getSensorsData()
        finger_pos_data = np.zeros(16,dtype=np.float64)
        finger_pressure_data = np.zeros(4, dtype=np.float64)
        finger_pos_data = self.sensors_data[12:28]
        finger_pressure_data = self.sensors_data[28:32]
        return finger_pos_data, finger_pressure_data

    def get_ur5e_data(self):
        self.getSensorsData()
        arm_pos_data = np.zeros(6, dtype=np.float64)
        arm_vel_data = np.zeros(6, dtype=np.float64)
        arm_pos_data = self.sensors_data[0:6]
        arm_vel_data = self.sensors_data[6:12]
        return arm_pos_data, arm_vel_data