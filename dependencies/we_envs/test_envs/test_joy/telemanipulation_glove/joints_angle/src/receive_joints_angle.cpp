#include "ros/ros.h"
#include "joints_angle/joints_angle.h"
#include <std_msgs/Float32.h>
#include <stdio.h>
#include <math.h>
#include <vector>
using namespace std;

#define PI 3.1415926

ros::Subscriber JointsAngleSub;
ros::Publisher JointsAnglePub;

ros::Publisher heheAnglePub;


 void onJointsAngle(const joints_angle::joints_angleConstPtr & cmd_msgs);

double  L[5][8];
double a[5];

double last_bending_angles[5][2]={0};

void read_handconfig(ros::NodeHandle private_nh)
{
    //Thumb
    private_nh.param("TL1",L[0][0], 0.0);
    private_nh.param("TL2",L[0][1], 0.0);
    private_nh.param("TL3",L[0][2], 0.0);
    private_nh.param("TL4",L[0][3], 0.0);
    private_nh.param("TL5",L[0][4], 0.0);
    private_nh.param("TL6",L[0][5], 0.0);
    private_nh.param("TL7",L[0][6], 0.0);
    private_nh.param("TL8",L[0][7], 0.0);
    private_nh.param("Ta3", a[0], 0.0);


    //ForeFinger
    private_nh.param("FL1",L[1][0], 0.0);
    private_nh.param("FL2",L[1][1], 0.0);
    private_nh.param("FL3",L[1][2], 0.0);
    private_nh.param("FL4",L[1][3], 0.0);
    private_nh.param("FL5",L[1][4], 0.0);
    private_nh.param("FL6",L[1][5], 0.0);
    private_nh.param("FL7",L[1][6], 0.0);
    private_nh.param("FL8",L[1][7], 0.0);
    private_nh.param("Fa3", a[1], 0.0);

    //MiddleFinger
    private_nh.param("ML1",L[2][0], 0.0);
    private_nh.param("ML2",L[2][1], 0.0);
    private_nh.param("ML3",L[2][2], 0.0);
    private_nh.param("ML4",L[2][3], 0.0);
    private_nh.param("ML5",L[2][4], 0.0);
    private_nh.param("ML6",L[2][5], 0.0);
    private_nh.param("ML7",L[2][6], 0.0);
    private_nh.param("ML8",L[2][7], 0.0);
    private_nh.param("Ma3", a[2], 0.0);

    //RingFinger
    private_nh.param("RL1",L[3][0], 0.0);
    private_nh.param("RL2",L[3][1], 0.0);
    private_nh.param("RL3",L[3][2], 0.0);
    private_nh.param("RL4",L[3][3], 0.0);
    private_nh.param("RL5",L[3][4], 0.0);
    private_nh.param("RL6",L[3][5], 0.0);
    private_nh.param("RL7",L[3][6], 0.0);
    private_nh.param("RL8",L[3][7], 0.0);
    private_nh.param("Ra3", a[3], 0.0);

    //LittleFinger
    private_nh.param("LL1",L[4][0], 0.0);
    private_nh.param("LL2",L[4][1], 0.0);
    private_nh.param("LL3",L[4][2], 0.0);
    private_nh.param("LL4",L[4][3], 0.0);
    private_nh.param("LL5",L[4][4], 0.0);
    private_nh.param("LL6",L[4][5], 0.0);
    private_nh.param("LL7",L[4][6], 0.0);
    private_nh.param("LL8",L[4][7], 0.0);
    private_nh.param("La3", a[4], 0.0);


}

int main(int argc, char **argv){
    ros::init(argc, argv, "recive_joints_angle");


    ros::NodeHandle nh("~");
    read_handconfig(nh);

    ros::NodeHandle n;
    JointsAngleSub = n.subscribe<joints_angle::joints_angle>("JointsAngle",1,&onJointsAngle);
    JointsAnglePub = n.advertise<joints_angle::joints_angle>("RealJointsAngle",1);

    heheAnglePub = n.advertise<std_msgs::Float32>("heheangle",1);

    ros::spin();
}

void getTheRealAngle(int index, float a1_input,float a2_input,float& theta2,float& theta1){


        float a1 = a1_input-1*PI/180.0;
        float a2 = a2_input+1*PI/180.0;

        float b1, b2, b5, P, Q, M;
        float a3 = a[index]*PI/180.0;
        float L1 = L[index][0];
        float L2 = L[index][1];
        float L3 = L[index][2];
        float L4 = L[index][3];
        float L5 = L[index][4];
        float L6 = L[index][5];
        float L7 = L[index][6];
        float L8 = L[index][7];

        float last_theta1 = last_bending_angles[index][0];
        float last_theta2 = last_bending_angles[index][1];

	float J, K, N;

	b1 = PI - a1;
	b2 = a2 - a3 - b1;

	P = -L3 + L5 * sin(b1) + L6 * sin(b2);
	Q = -L4 - L5 * cos(b1) + L6 * cos(b2);
	M = (P*P + Q * Q + L2 * L2 - L1 * L1) / (2 * L2);


        float theta2_sin_1 = (2*M*Q + sqrt(4 * pow(M*Q, 2) - 4 * (Q*Q + P * P)*(M*M - P * P))) / (2 * (Q*Q + P * P));

        if(index==0)
        {  //hehe, not know why yet
            theta2_sin_1 = (2*M*Q - sqrt(4 * pow(M*Q, 2) - 4 * (Q*Q + P * P)*(M*M - P * P))) / (2 * (Q*Q + P * P));
        }


        float theta2_1 = asin(theta2_sin_1);

        // we accept if theta_2 slightly bigger than 90 degree
        float cos_theta2_1 = (M-Q*sin(theta2_1));
        //ROS_INFO("P=%0.3f",P);
        if(cos_theta2_1<0.0)
        {
            theta2_1=PI-theta2_1;
        }

         if(isnan(theta2_1)) theta2_1=last_theta2;


        //solve theta1
        float angle_DC;
        if (fabs(L3+L2*cos(theta2_1)-L5*sin(b1)-L6*sin(b2)) < 0.01)
        {
            angle_DC = PI/2;
        }
        else {
            angle_DC=atan2(L4+L2*sin(theta2_1)+L5*cos(b1)-L6*cos(b2), L3+L2*cos(theta2_1)-L5*sin(b1)-L6*sin(b2));
            if (index==0) //if is the thumb
            {
                if (angle_DC>=PI/2 && angle_DC<=PI)
                {
                    angle_DC = angle_DC-PI;
                }
                else if(angle_DC<=0 && angle_DC>=-PI)
                {
                    angle_DC = PI+angle_DC;
                }
            }
            else { //if not the thumb
                if(angle_DC<0) angle_DC+=2*PI;
                    angle_DC-=PI;
            }

        }

        float total_angle = angle_DC+atan(L8/L7);
        theta1 = total_angle - theta2_1;

//        if(index==0)
//               {
//                   ROS_WARN("total_angle: %0.2f", total_angle/PI*180);

//                   ROS_INFO("--------------");
//                   std_msgs::Float32 hehemsg;
//                   hehemsg.data=total_angle/PI*180;
//                   heheAnglePub.publish(hehemsg);
//               }

        last_bending_angles[index][0]=theta1;
        last_bending_angles[index][1]=theta2_1;

        theta2 = theta2_1 / PI * 180;
        theta1 = theta1 / PI * 180;
}



void onJointsAngle(const joints_angle::joints_angleConstPtr & cmd_msgs){
    joints_angle::joints_angle realAngle;
    getTheRealAngle(0, cmd_msgs->T2*PI/180.0,cmd_msgs->T1*PI/180.0,(realAngle.T2),(realAngle.T1));
    realAngle.T3 = cmd_msgs->T3;
    realAngle.T4 = cmd_msgs->T4;
    realAngle.T5 = cmd_msgs->T5;
    getTheRealAngle(1, cmd_msgs->F2*PI/180.0,cmd_msgs->F1*PI/180.0,(realAngle.F2),(realAngle.F1));
    realAngle.F3 = cmd_msgs->F3;
    realAngle.F4 = cmd_msgs->F4;
    getTheRealAngle(2, cmd_msgs->M2*PI/180.0,cmd_msgs->M1*PI/180.0,(realAngle.M2),(realAngle.M1));
    realAngle.M3 = cmd_msgs->M3;
    realAngle.M4 = cmd_msgs->M4;
    getTheRealAngle(3, cmd_msgs->R2*PI/180.0,cmd_msgs->R1*PI/180.0,(realAngle.R2),(realAngle.R1));
    realAngle.R3 = cmd_msgs->R3;
    realAngle.R4 = cmd_msgs->R4;

    JointsAnglePub.publish(realAngle);
}
