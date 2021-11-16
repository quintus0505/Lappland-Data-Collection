import numpy as np
from numpy import linalg

import cmath
import math
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

global mat
mat = np.matrix

# ****** Coefficients ******


global d1, a2, a3, a7, d4, d5, d6
d1 = 0.1625
a2 = -0.425
a3 = -0.3922

d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

a7 = 0.1817

global d, a, alph

d = mat([0.1625, 0, 0, 0.1333, 0.0997, 0.0996]) #ur5e
# d = mat([0.1273, 0, 0, 0.163941, 0.1157, 0.0922])  # ur10 mm
a =mat([0 ,-0.425 ,-0.3922 ,0 ,0 ,0]) #ur5e
# a = mat([0, -0.612, -0.5723, 0, 0, 0])  # ur10 mm
alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  #ur5e
# alph = mat([pi / 2, 0, 0, pi / 2, -pi / 2, 0])  # ur10


# ************************************************** FORWARD KINEMATICS

def AH(n, th, c):
    T_a = mat(np.identity(4), copy=False)
    T_a[0, 3] = a[0, n - 1]
    T_d = mat(np.identity(4), copy=False)
    T_d[2, 3] = d[0, n - 1]

    Rzt = mat([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
               [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], copy=False)

    Rxa = mat([[1, 0, 0, 0],
               [0, cos(alph[0, n - 1]), -sin(alph[0, n - 1]), 0],
               [0, sin(alph[0, n - 1]), cos(alph[0, n - 1]), 0],
               [0, 0, 0, 1]], copy=False)

    A_i = T_d * Rzt * T_a * Rxa

    return A_i


def HTrans(th, c):
    A_1 = AH(1, th, c)
    A_2 = AH(2, th, c)
    A_3 = AH(3, th, c)
    A_4 = AH(4, th, c)
    A_5 = AH(5, th, c)
    A_6 = AH(6, th, c)

    T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6

    return T_06

def URbase2rosbase(T_06):
    static_tf =mat(np.identity(4), copy=False)
    static_tf[0,0]=-1.0
    static_tf[0,1]=0.0
    static_tf[1,0]=0.0
    static_tf[1,1]=-1.0

    return static_tf*T_06


# ************************************************** INVERSE KINEMATICS

def invKine(desired_pos):  # T60
    th = mat(np.zeros((6, 8)))
    P_05 = (desired_pos * mat([0, 0, -d6, 1]).T - mat([0, 0, 0, 1]).T)

    # **** theta1 ****

    psi = atan2(P_05[2 - 1, 0], P_05[1 - 1, 0])
    phi = acos(d4 / sqrt(P_05[2 - 1, 0] * P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0]))
    # The two solutions for theta1 correspond to the shoulder
    # being either left or right
    th[0, 0:4] = pi / 2 + psi + phi
    th[0, 4:8] = pi / 2 + psi - phi
    th = th.real

    # **** theta5 ****

    cl = [0, 4]  # wrist up or down
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = T_10 * desired_pos
        th[4, c:c + 2] = + acos((T_16[2, 3] - d4) / d6);
        th[4, c + 2:c + 4] = - acos((T_16[2, 3] - d4) / d6);

    th = th.real

    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = linalg.inv(T_10 * desired_pos)
        th[5, c:c + 2] = atan2((-T_16[1, 2] / sin(th[4, c])), (T_16[0, 2] / sin(th[4, c])))

    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = AH(6, th, c)
        T_54 = AH(5, th, c)
        T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0, 0, 0, 1]).T
        t3 = cmath.acos((linalg.norm(P_13) ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3))  # norm ?
        th[2, c] = t3.real
        th[2, c + 1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = linalg.inv(AH(6, th, c))
        T_54 = linalg.inv(AH(5, th, c))
        T_14 = (T_10 * desired_pos) * T_65 * T_54
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0, 0, 0, 1]).T

        # theta 2
        th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3 * sin(th[2, c]) / linalg.norm(P_13))
        # theta 4
        T_32 = linalg.inv(AH(3, th, c))
        T_21 = linalg.inv(AH(2, th, c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
    th = th.real

    return th

def quat2mat(quat):
  """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
  quat = np.asarray(quat, dtype=np.float64)
  assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

  w, x, y, z = quat[0], quat[1], quat[2], quat[3]
  Nq = np.sum(quat * quat, axis=-1)
  s = 2.0 / Nq
  X, Y, Z = x * s, y * s, z * s
  wX, wY, wZ = w * X, w * Y, w * Z
  xX, xY, xZ = x * X, x * Y, x * Z
  yY, yZ, zZ = y * Y, y * Z, z * Z

  mat = np.empty((3, 3), dtype=np.float64)
  mat[0, 0] = 1.0 - (yY + zZ)
  mat[0, 1] = xY - wZ
  mat[0, 2] = xZ + wY
  mat[1, 0] = xY + wZ
  mat[1, 1] = 1.0 - (xX + zZ)
  mat[1, 2] = yZ - wX
  mat[2, 0] = xZ - wY
  mat[2, 1] = yZ + wX
  mat[2, 2] = 1.0 - (xX + yY)
  return np.matrix(mat)



def direct_ik(position, quat, qpos_last):
  '''
  first, transform mujoco frame into UR frame

  '''

  matrix_ee = quat2mat(quat)
  matrix_z = np.matrix('-1 0 0; 0 -1 0; 0 0 1')
  rotation_matrix = np.dot(matrix_z, matrix_ee)


  # rotation_matrix = euler2mat(euler_pose)
  np_pose = np.zeros((4,4),dtype=np.float64)
  np_pose[:3, :3]=rotation_matrix
  for i in range(2):
    np_pose[i][3]=-position[i]
  np_pose[2][3]=position[2]
  np_pose[3][3]=1.0

  np_pose_input = np.matrix(np_pose)

  target_q=invKine(np_pose_input)
  return target_q.copy()


def direct_ik_2(position, quat):
  '''
  first, transform mujoco frame into UR frame
  '''

  matrix_ee = quat2mat(quat)
  matrix_z = np.matrix('-1 0 0; 0 -1 0; 0 0 1')
  rotation_matrix = np.dot(matrix_z, matrix_ee)


  # rotation_matrix = euler2mat(euler_pose)
  np_pose = np.zeros((4,4),dtype=np.float64)
  np_pose[:3, :3]=rotation_matrix
  for i in range(2):
    np_pose[i][3]=-position[i] # the x,y axis of two  frame is opposite
  np_pose[2][3]=position[2]-0.8 # because the door height is 0.8m
  np_pose[3][3]=1.0

  np_pose_input = np.matrix(np_pose)

  target_q=invKine(np_pose_input)
  return target_q.copy()