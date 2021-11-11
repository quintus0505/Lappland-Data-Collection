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
from typing import Dict, List
import we_envs
import copy
from abc import ABC
import abc
from primitive.Primitive import Primitive, ApproachPrimitive, Move2targetPrimitive, GraspPrimitive
from primitive.Primitive import DoorApproachPrimitive, DoorGraspLatchPrimitive, DoorOpenPrimitive
from primitive.Primitive import HammerApproachToolPrimitive, HammerApproachNailPrimitive, HammerNailGoInsidePrimitive


class StateMachine():
    def __init__(self, N=3):
        self.state = 0
        self.state_N = N

    def update_state(self):
        if self.state == self.state_N - 1:
            return
        else:
            self.state += 1
        return

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state = 0
        return


class TaskDemoCollector():
    def __init__(self, primitives: List[Primitive], state_machine: StateMachine, env):
        self.primitives = primitives
        self.primitives_name = []
        self.state_machine = state_machine  # TODO: state_machine describe the process, future change to graph-structure
        self.primitive_count = len(primitives)
        self.env_name = str(env.spec.id)
        self.env = env
        self.pi = None

    def set_policy(self, pi):
        self.pi = pi

    def debug(self):
        # for _ in range(30):
        #     self.env.render()
        # print(self.env.obj2target())
        pass

    def finish_single_traj(self, traj_success=True):
        # print(self.primitives[0].current_trajectory['begin_env_state']['target_pos'])
        valid = traj_success
        for primitive in self.primitives:
            if primitive.demo_collector.current_trajectory['begin_env_state'] is None:
                valid = False
            elif not primitive.evaluate_success(primitive.demo_collector.current_trajectory):  # TODO: why?
                valid = False
            if len(primitive.demo_collector.current_trajectory['state']) < 5:
                valid = False
            # if primitive.demo_collector.current_trajectory['end_env_state'] is None:
            #     valid = False

        if valid is False:
            for primitive in self.primitives:
                primitive.demo_collector.reset_current_traj()
        else:
            for primitive in self.primitives:
                primitive.demo_collector.current_trajectory['state'] = np.array(
                    primitive.demo_collector.current_trajectory['state'])
                primitive.demo_collector.current_trajectory['action'] = np.array(
                    primitive.demo_collector.current_trajectory['action'])
                # primitive.demo_collector.current_trajectory['end_env_state'] = np.array(
                #     primitive.demo_collector.current_trajectory['end_env_state'])
                # primitive.demo_collector.current_trajectory['begin_env_state'] = np.array(
                #     primitive.demo_collector.current_trajectory['begin_env_state'])
                primitive.demo_collector.current_trajectory['goal'] = np.array(
                    primitive.demo_collector.current_trajectory['goal'])

                primitive.demo_collector.demo_trajs.append(primitive.demo_collector.current_trajectory)
                primitive.demo_collector.reset_current_traj()
        return valid

    def reset(self):

        for primitive in self.primitives:
            primitive.demo_collector.reset()
        self.state_machine.reset()
        env_state = self.env.reset()
        return env_state

    def auto_collect_traj(self, num_episodes=20, with_image=False, with_traingle_images=False):

        self.state_machine.reset()
        self.primitives_name = ['Approach', 'Grasp', 'Move2Target']
        horizon = self.env._max_episode_steps
        s = np.array(self.env.reset())
        current_episode = 0
        watch_dog = 0

        while current_episode < num_episodes:
            print("collect: ", current_episode)
            watch_dog += 1
            if watch_dog >= 5 * num_episodes:
                print("trajectories are not all collected !")
                break
            done = False
            episode_step = 0
            primitive_step_log = 0
            s = self.reset()
            self.primitives[self.state_machine.state].demo_collector.begin_information(env=self.env)

            while not done:

                a = self.pi(s)
                episode_step += 1

                self.primitives[self.state_machine.state].demo_collector.record_state_action_pair(state=s, action=a)
                self.primitives[self.state_machine.state].demo_collector.record_fullstate(
                    fullstate=self.env.get_env_state())
                # TODO: exist bugs when the environment don't have get_obs_same_dim()
                self.primitives[self.state_machine.state].demo_collector.record_state_same_dim(
                    state_same_dim=self.env.get_obs_same_dim())
                if with_image:
                    if not with_traingle_images:
                        self.primitives[self.state_machine.state].demo_collector.record_images(
                            images=self.env.get_images())
                    if with_traingle_images:
                        self.primitives[self.state_machine.state].demo_collector.record_images(
                            images=self.env.get_images_traingle_cameras())

                s, r, _, info = self.env.step(a)
                # self.env.render()
                self.debug()

                if self.primitives[self.state_machine.state].leave_condition(env=self.env, state=s, action=a):
                    # print('leave primitive: ', self.state_machine.state)
                    full_state = self.env.get_env_state()
                    hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], \
                                                               full_state['target_pos'], full_state['palm_pos']
                    qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']

                    # update the state-machine and print the next primitive name
                    palm_and_obj = np.append(palm_pos, obj_pos)
                    target_and_obj = np.append(target_pos, obj_pos)
                    if self.state_machine.state == 0:
                        for j in range(episode_step - primitive_step_log):
                            self.primitives[self.state_machine.state].demo_collector.add_goal(obj_pos)
                            self.primitives[self.state_machine.state].demo_collector.add_primitive_label(
                                self.primitives_name[self.state_machine.state])
                            # print(palm_pos)
                    elif self.state_machine.state == 1:
                        for j in range(episode_step - primitive_step_log):
                            self.primitives[self.state_machine.state].demo_collector.add_goal(obj_pos)
                            self.primitives[self.state_machine.state].demo_collector.add_primitive_label(
                                self.primitives_name[self.state_machine.state])
                        # print("palm: {} obj : {} dist : {}".format(palm_pos, obj_pos, np.linalg.norm(palm_pos-obj_pos)))

                    elif self.state_machine.state == 2:
                        for j in range(episode_step - primitive_step_log):
                            self.primitives[self.state_machine.state].demo_collector.add_goal(target_pos)
                            self.primitives[self.state_machine.state].demo_collector.add_primitive_label(
                                self.primitives_name[self.state_machine.state])

                    self.primitives[self.state_machine.state].demo_collector.end_information(env=self.env)
                    primitive_step_log = episode_step

                    self.state_machine.update_state()
                    self.primitives[self.state_machine.state].demo_collector.begin_information(env=self.env)

                done = episode_step >= horizon or self.primitives[2].leave_condition(env=self.env)

            traj_success = self.finish_single_traj(episode_step < horizon)
            if traj_success:
                current_episode += 1

        self.env.close()
        return current_episode

    def save_demonstrations(self, given_fname='', mpi_rank=None, mpi_dir=''):
        for primitive in self.primitives:
            primitive.demo_collector.save_demonstrations(env_name=self.env_name, primitive_name=primitive.name,
                                                         given_fname=given_fname,
                                                         mpi_rank=mpi_rank, mpi_dir=mpi_dir)
        print("demonstrations collected!")


class RelocateDemoCollector(TaskDemoCollector):

    def __init__(self, env_name):
        env = gym.make(env_name)
        approach_primitive = ApproachPrimitive(env)
        grasp_primitive = GraspPrimitive(env)
        move2target_primitive = Move2targetPrimitive(env)
        self.primitives = [approach_primitive, grasp_primitive, move2target_primitive]
        self.primitives_name = ['Approach', 'Grasp', 'Move2Target']
        self.state_machine = StateMachine(N=len(self.primitives))

        super(RelocateDemoCollector, self).__init__(self.primitives, self.state_machine, env=env)
