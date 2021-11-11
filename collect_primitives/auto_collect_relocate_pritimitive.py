import we_envs
import click 
import os
import sys
sys.path.append("..")
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv
import time
from tasks.Task_relocate import TaskDemoCollector, RelocateDemoCollector

from primitive.user_config import PRIMITIVE_DEMONSTRATIONS_PATH
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''

env_name = 'relocate-v6'

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--policy', type=str, default='../policies/')
@click.option('--num_episodes', type=int, default=5)
@click.option('--with_image', type=bool, default=False)
@click.option('--with_traingle_images', type=bool, default=False) # whether use camera_middle
def main(policy, num_episodes, with_image, with_traingle_images):
    global env_name
    policy = policy + env_name+ '.pickle'
    env_name = 'Adroit-' + env_name
    pi_load = pickle.load(open(policy, 'rb'))
    pi = lambda x: pi_load.get_action(x)[1]['evaluation']

    if mpi_rank==0:
        if not os.path.exists(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj'):
            os.makedirs(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj')

    demo_collector =RelocateDemoCollector(env_name=env_name)
    demo_collector.set_policy(pi)
    demo_collector.auto_collect_traj(num_episodes=num_episodes, with_image=with_image, with_traingle_images=with_traingle_images)
    demo_collector.save_demonstrations(mpi_rank=mpi_rank, mpi_dir='multiple_traj/')


def relocate_visualize_primitive_demo(speed=1):
    from spinup.demonstrations.collect_utils.dapg.collect_primitives.collect_utils import visualize_primitive_demo
    primitive_name ='Move2Target'   # Approach Grasp Move2Target
    global env_name
    visualize_primitive_demo(env_name=env_name, primitive_name=primitive_name, speed=speed)


def relocate_combine_mpi_trajectories(): # for mpi multiple trajectories
    from spinup.demonstrations.collect_utils.dapg.collect_primitives.collect_utils import combine_mpi_trajectories
    primitive_names =['Approach', 'Grasp', 'Move2Target']
    combine_mpi_trajectories(env_name=env_name, primitive_names=primitive_names)



if __name__ == '__main__':
    # main()
    # relocate_combine_mpi_trajectories()
    relocate_visualize_primitive_demo()
