import we_envs
import click
import os
import gym
import sys
import numpy as np
import pickle
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from utils.user_config import PRIMITIVE_DEMONSTRATIONS_PATH, POLICIES_PATH
from mjrl.utils.gym_env import GymEnv
import time
from primitive.Primitive import Primitive, HammerApproachToolPrimitive, \
    HammerApproachNailPrimitive, HammerNailGoInsidePrimitive
from primitive.Primitive import DoorApproachPrimitive, DoorGraspLatchPrimitive, \
    DoorOpenPrimitive
from primitive.Primitive import ApproachPrimitive, Move2targetPrimitive,GraspPrimitive

from utils.user_config import PRIMITIVE_DEMONSTRATIONS_PATH
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()



@click.option('--speed', type=float, default=5.0)
def visualize_primitive_demo(env_name, primitive_name, fpath='', speed=1,  Rank=None):
    env_name = 'Adroit-' + env_name
    env = gym.make(env_name)
    # print(str(env.spec.id))

    if not fpath=='':
        fname = os.path.join(fpath, str(env_name)+'_'+str(primitive_name) + '.pickle')
    else:
        # primitive_name = 'HammerNailGoInside' # 'HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside'
        current_file_path = os.path.abspath(os.path.dirname(__file__))
        if Rank is None:
            fname = PRIMITIVE_DEMONSTRATIONS_PATH+str(env_name)+'_'+str(primitive_name) + '.pickle'
        else:
            fname = PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj/'+str(env_name)+'_'+str(primitive_name) + '_' + Rank + '.pickle'
        fname = os.path.join(current_file_path, fname)
        # demo = pickle.load(open(fname, 'rb'))

    primitive = None
    if primitive_name=='Approach':
        primitive = ApproachPrimitive(env=env)
    elif primitive_name=='Grasp':
        primitive = GraspPrimitive(env=env)
    elif primitive_name=='Move2Target':
        primitive = Move2targetPrimitive(env=env)

    if primitive_name == 'DoorApproach':
        primitive = DoorApproachPrimitive(env=env)
    elif primitive_name == 'DoorGraspLatch':
        primitive = DoorGraspLatchPrimitive(env=env)
    elif primitive_name == 'DoorOpen':
        primitive = DoorOpenPrimitive(env=env)

    if primitive_name == 'HammerApproachTool':
        primitive = HammerApproachToolPrimitive(env=env)
    elif primitive_name == 'HammerApproachNail':
        primitive = HammerApproachNailPrimitive(env=env)
    elif primitive_name == 'HammerNailGoInside':
        primitive = HammerNailGoInsidePrimitive(env=env)
    primitive.load_demos(fname)

    primitive.visualize_demos(speed=1.0)
    # primitive.visualize_image(trajectory_index=0, frame_index=0, camera_order=2)

from os import walk
def combine_mpi_trajectories(env_name, primitive_names): # for mpi multiple trajectories
    # env_name = 'Adroit-' + env_name
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    # primitive_names =['HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside']

    for primitive_name in primitive_names:
        total_trajectories = []
        fname_prefix = str(env_name) + '_' + str(primitive_name) + '_'
        for (dirpath, dirnames, filenames) in walk(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj/'):
            for filename in filenames:
                if fname_prefix in filename:
                    fname = os.path.join(PRIMITIVE_DEMONSTRATIONS_PATH+ 'multiple_traj/', filename)
                    demo = pickle.load(open(fname, 'rb'))
                    total_trajectories += demo

        combined_file_name = PRIMITIVE_DEMONSTRATIONS_PATH  + str(env_name) + '_' + str(primitive_name) + '.pickle'

        with open(combined_file_name, 'wb') as f:
            pickle.dump(total_trajectories, f)
            print(str(env_name) + ' ' + str(primitive_name) + " combined demonstrations collected!")

def get_image_data(env, camera_name):
    env = env.unwrapped
    # env.reset()
    if env.viewer is None:
        env.mj_viewer_setup()
    image_data = env.sim.render(width=512, height=512, camera_name=camera_name, depth=False)
    return image_data

import matplotlib.pyplot as plt
def draw_image(image_data):
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # combine_mpi_trajectories()
    # visualize_primitive_demo()
    env_name = 'relocate-v6'
    env_name = 'Adroit-' + env_name
    env = gym.make(env_name)
    # image_data = get_image_data(env=env, camera_name='camera_left')
    env = env.unwrapped
    images_data = env.get_images()
    draw_image(images_data[0])


