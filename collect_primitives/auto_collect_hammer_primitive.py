import click
import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import pickle

from tasks.Task_hammer import HammerDemoCollector

from utils.user_config import PRIMITIVE_DEMONSTRATIONS_PATH, POLICIES_PATH

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


DESC = '''
Helper script to use this code.\n
USAGE:\n
    Collects data\n
    $ python collect_primitives/auto_collect_hammer_primitive.py --option collect
    Visualizes policy on the env\n
    $ python collect_primitives/auto_collect_hammer_primitive.py --option visualize *\n
    (choose from ['HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside']
'''
env_name = 'hammer-v6'

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--policy', type=str, default=POLICIES_PATH)
@click.option('--num_episodes', type=int, default=50)
@click.option('--with_image', type=bool, default=False)
@click.option('--option', type=click.Choice(['collect', 'visualize']), default="collect")
@click.option('--primitive_name', type=click.Choice(['HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside']), default="HammerApproachTool")

def main(policy, num_episodes, with_image, option, primitive_name):
    if option == "collect":
        global env_name
        policy = policy + env_name + '.pickle'
        env_name = 'Adroit-' + env_name
        pi_load = pickle.load(open(policy, 'rb'))
        pi = lambda x: pi_load.get_action(x)[1]['evaluation']

        if mpi_rank==0:
            if not os.path.exists(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj'):
                os.makedirs(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj')

        demo_collector = HammerDemoCollector(env_name=env_name)
        demo_collector.set_policy(pi)
        demo_collector.auto_collect_traj(num_episodes=num_episodes, with_image=with_image)
        demo_collector.save_demonstrations(mpi_rank=mpi_rank, mpi_dir='multiple_traj/')
        mpi_comm.Barrier()
        if mpi_rank == 0:
            hammer_combine_mpi_trajectories()
            if os.path.exists(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj'):
                shutil.rmtree(PRIMITIVE_DEMONSTRATIONS_PATH + 'multiple_traj')
    else:
        hammer_visualize_primitive_demo(primitive_name=primitive_name)

def hammer_visualize_primitive_demo(primitive_name, speed=1,):
    from utils.collect_utils import visualize_primitive_demo
    # primitive_name = 'HammerNailGoInside' # 'HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside'
    global env_name
    visualize_primitive_demo(env_name=env_name, primitive_name=primitive_name, speed=speed)


def hammer_combine_mpi_trajectories(): # for mpi multiple trajectories
    from utils.collect_utils import combine_mpi_trajectories
    primitive_names =['HammerApproachTool', 'HammerApproachNail', 'HammerNailGoInside']
    combine_mpi_trajectories(env_name=env_name, primitive_names=primitive_names)



if __name__ == '__main__':
    main()
