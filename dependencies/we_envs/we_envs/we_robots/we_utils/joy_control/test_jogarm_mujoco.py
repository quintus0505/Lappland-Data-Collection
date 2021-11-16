#!/usr/bin/env python
# coding:utf-8
import rospy
from sensor_msgs.msg import Joy
from  geometry_msgs.msg import TwistStamped
import threading,time
from std_msgs.msg import  Float64
from std_msgs.msg import  Bool

scale = 0.1
scale_angle = 0.5
lock = threading.Lock()

control_pressure = Float64()
control_pressure.data = 0.0

open_pressure = Float64()
open_pressure.data = 70.0


gripper_close_flag = False
gripper_close = Bool()
gripper_close.data = True

twist = TwistStamped()
twist.twist.linear.x = 0
twist.twist.linear.y = 0
twist.twist.linear.z = 0
twist.twist.angular.x = 0
twist.twist.angular.y = 0
twist.twist.angular.z = 0


gripper_angle = 0.0
gripper_angle_scale=0.2
gripper_max_angle = 0.08
gripper_min_angle = -0.08

def callback(velocity):
    axes = velocity.axes
    buttons = velocity.buttons
    v = [-axes[0]* scale, axes[1]* scale ,0.0]
    w = [-axes[3]*scale_angle,0.0, -axes[2]*scale_angle]
    if (buttons[4]==1 and buttons[5]==1):
        pass
    elif (buttons[4]==1):
        v[2]=scale
    elif (buttons[5]==1):
        v[2]=-scale

    if (buttons[6]==1 and buttons[7]==1):
        pass
    elif (buttons[6]==1):
        w[1]=scale_angle
    elif (buttons[7]==1):
        w[1]=-scale_angle

    global  twist
    lock.acquire()
    twist.header = velocity.header
    twist.twist.linear.x = v[0]
    twist.twist.linear.y = v[1]
    twist.twist.linear.z = v[2]
    twist.twist.angular.x = w[0]
    twist.twist.angular.y = w[1]
    twist.twist.angular.z = w[2]
    lock.release()

    global gripper_angle
    if (buttons[3] == 1):
        gripper_angle = gripper_angle + gripper_angle_scale
        if gripper_angle > gripper_max_angle:
            gripper_angle = gripper_max_angle
        gripper_angel_msg = Float64()
        gripper_angel_msg.data = gripper_angle
        gripper_pub.publish(gripper_angle)

    if (buttons[2] == 1):
        gripper_angle = gripper_angle - gripper_angle_scale
        if gripper_angle < gripper_min_angle:
            gripper_angle = gripper_min_angle
        gripper_angel_msg = Float64()
        gripper_angel_msg.data = gripper_angle
        gripper_pub.publish(gripper_angle)

    gripper_angle_increasement_msg = Float64()

    if (buttons[3]==1):
        gripper_angle_increasement_msg.data = gripper_angle_scale
        gripper_increasement_pub.publish(gripper_angle_increasement_msg)
    if (buttons[2]==1):
        gripper_angle_increasement_msg.data = -gripper_angle_scale
        gripper_increasement_pub.publish(gripper_angle_increasement_msg)



def publisher_thread(event):
    pub.publish(twist)


def listener():
    global  pub
    global gripper_pub, gripper_increasement_pub

    rospy.init_node('joy_listener_mujoco', anonymous=True)
    pub = rospy.Publisher('velocity', TwistStamped, queue_size=1)
    gripper_pub = rospy.Publisher('gripper_pos', Float64, queue_size=1)
    gripper_increasement_pub = rospy.Publisher('gripper_pos_increasement', Float64, queue_size=1)
    rospy.Subscriber("joy", Joy, callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.Timer(rospy.Duration(0.01), publisher_thread)
    rospy.spin()


if __name__ == '__main__':
    listener()
