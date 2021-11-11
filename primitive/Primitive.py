import we_envs
import click
import os
import sys
sys.path.append("..")
import gym
import numpy as np
import pickle
# from mjrl.utils.gym_env import GymEnv
import time
from typing import Dict, List
import we_envs
import copy
import torch
from abc import ABC
import abc
from constraint.Constraint import Constraint, ConstraintManager
from primitive.user_config import PRIMITIVE_DEMONSTRATIONS_PATH
import matplotlib.pyplot as plt


class PrimitiveDemoCollector():
    def __init__(self):
        self.demo_trajs = []
        self.current_trajectory = self.reset_current_traj()
        self.visited = False

    def reset_current_traj(self):
        self.current_trajectory = dict()
        self.current_trajectory['begin_env_state'] = None
        self.current_trajectory['end_env_state'] = None
        self.current_trajectory['goal'] = []
        self.current_trajectory['label'] = []
        self.current_trajectory['state'] = []
        self.current_trajectory['action'] = []

        self.current_trajectory['full_state'] = []
        self.current_trajectory['state_same_dim'] = []
        self.current_trajectory['images']=[]
        return self.current_trajectory

    def begin_information(self, env):
        if not self.visited:
            begin_state = env.get_env_state()
            # print("palm: {} obj : {}".format(begin_state['palm_pos'], begin_state['obj_pos']))
            self.record_begin_env_state(state=begin_state)
            self.visited = True

    def record_begin_env_state(self, state):
        self.current_trajectory['begin_env_state'] = copy.deepcopy(state)

    def record_state_action_pair(self, state, action):
        self.current_trajectory['state'].append(state)
        self.current_trajectory['action'].append(action)

    def record_fullstate(self, fullstate):
        self.current_trajectory['full_state'].append(fullstate)

    def record_state_same_dim(self, state_same_dim):
        self.current_trajectory['state_same_dim'].append(state_same_dim)

    def record_images(self, images):
        self.current_trajectory['images'].append(images)

    def add_goal(self, state):
        if not state == []:
            self.current_trajectory['goal'].append(copy.deepcopy(state))
            # print(self.current_trajectory['goal'])
        else:
            print(state)

    def add_primitive_label(self, primitive_name):
        if primitive_name != '':
            self.current_trajectory['label'].append(copy.deepcopy(primitive_name))
            # print(self.current_trajectory['goal'])

    def end_information(self, env):
        end_state = env.get_env_state()
        self.record_end_env_state(state=end_state)

    def record_end_env_state(self, state):
        self.current_trajectory['end_env_state'] = copy.deepcopy(state)
        # self.demo_trajs.append(self.current_trajectory)
        # self.reset_current_traj()

    def get_demonstrations(self):
        return self.demo_trajs

    def save_demonstrations(self, env_name, primitive_name, given_fname='', mpi_rank=None, mpi_dir='', save_mode='wb'):
        if given_fname!='':
            fname = given_fname + str(env_name) + '_' + str(primitive_name) + '.pickle'
        else:
            if mpi_rank is None:
                fname = PRIMITIVE_DEMONSTRATIONS_PATH + str(env_name) + '_' + str(primitive_name) + '.pickle'
            else:
                fname = PRIMITIVE_DEMONSTRATIONS_PATH + mpi_dir+ str(env_name) + '_' + str(primitive_name)+ '_' + str(mpi_rank) + '.pickle'

        fpath = (str(fname)).rsplit("/",1)[0]
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        with open(fname, save_mode) as f:
            pickle.dump(self.demo_trajs, f)
            print(str(env_name) + ' ' + str(primitive_name) + "demonstrations collected!")

    def reset(self):
        self.reset_current_traj()
        # self.demo_trajs=[]
        self.visited = False

    def visualize_demo(self, env, demos, speed=1.0):
        env = env.unwrapped
        env.reset()
        print('demo count:', len(demos))
        for demo in demos:
            # env.reset()
            begin_state, end_state = demo['begin_env_state'], demo['end_env_state']
            state_seq, action_seq = demo['state'], demo['action']
            env.set_env_state(state_dict=begin_state)

            fps = int(10 / speed)
            fps = fps if fps > 0 else 1
            for action in action_seq:
                env.step(action)
                for _ in range(fps):
                    env.render()





class Primitive():
    def __init__(self, env, name, goal_dim=0, observation_dim=0, action_dim=0):
        self.env = env  # the environment to train the primitive, not the planner environment!
        self.env_name = str(env.spec.id)
        self.name = name
        self.horizon = 0  # max steps to be executed
        self.demo_collector = PrimitiveDemoCollector()

        self.constraint = ConstraintManager()
        self.TimeLimit, self.StateLimit = Constraint(name='time_limit'), Constraint(name='state_limit')
        self.ActionLimit, self.SafetyLimit = Constraint(name='action_limit'), Constraint(name='safety_limit')
        self.constraint.add_constraint(self.TimeLimit)
        self.constraint.add_constraint(self.StateLimit)
        self.constraint.add_constraint(self.ActionLimit)
        self.constraint.add_constraint(self.SafetyLimit)

        # self.policy = dict()    # pi, id, value-function, etc self.policy should be dict
        # self.policy['pi'] = None
        # self.policy['id'] = None
        # self.policy['v'] = None

        self.ac = None  # contain polic ac.pi ac.id ac.v
        self.primitive_id = -1

        self.goal = None  # goal-conditioned policy
        self.objective = None  # the objective to optimize the policy

        self.SVDD = None

        self.goal_dim = goal_dim
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # follow variables are used in one-shot imitation learning, to finetune the extracted goal
        self.inital_demo_goal = None
        # self.goal_opt_function = None
        self.one_shot_demo_states = [] # the one-shot demonstration, only include state sequence
        self.goal_pbounds = None
        self.goal_meaning = ''

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        self.one_shot_demo_states = one_shot_demonstration_states
        self.goal_pbounds = None
        self.inital_demo_goal = None

    def __str__(self):
        return self.name

    # def init_SVDD(self):
    #     self.SVDD = DeepSVDD(primitive_name=self.name)
    #     self.SVDD.set_net(obs_dim=self.observation_dim, goal_dim=self.goal_dim)
    #
    # def set_SVDD(self):
    #     self.SVDD = DeepSVDD(primitive_name=self.name)
    #     self.SVDD.load_data(self.name)
    #     self.SVDD.set_net(obs_dim=self.observation_dim, goal_dim=self.goal_dim)
    #     self.SVDD.train()
    #     print("train Primitive {name} SVDD finish!".format(name=self.name))

    # judge whether begin-constarint satisfied, allow to enter the primitive
    def enter_condition(self, env, state=None, action=None):
        raise NotImplementedError('Not implemented primitive!')

    # judge whether end-constarint satisfied, allow to leave the primitive
    def leave_condition(self, env, state=None, action=None):
        raise NotImplementedError('Not implemented primitive!')

    def load_demos(self, DemoFilePath):
        demo = pickle.load(open(DemoFilePath, 'rb'))
        self.demo_collector.demo_trajs = demo
        return len(demo)

    def load_policy(self, PolicyFilePath):
        # xxxx = load_polciy(PolicyFilePath)
        # self.policy = xxxx
        pass

    def visualize_demos(self, speed=1.0):
        print("demo trajs count:", len(self.demo_collector.demo_trajs))
        self.demo_collector.visualize_demo(env=self.env, demos=self.demo_collector.demo_trajs, speed=speed)

    def visualize_image(self, trajectory_index = 0, frame_index=1, camera_order=0):
        """
        visualize a certain image
        :param trajectory_index: which trajectory
        :param frame_index: the frame index  in the trajectory
        :param camera_order: which camera-image to be shown, if offered multi-view cameras
        :return:
        """
        demo = self.demo_collector.demo_trajs[trajectory_index]
        images = demo['images']
        select_image = images[frame_index][camera_order]
        plt.imshow(select_image)
        plt.axis('off')
        plt.show()

    def set_current_primitive_horizon(self, horizon):
        self.horizon = horizon

    def one_shot_finetune(self, one_shot_demo, env, pi_lr=1e-4, id_lr=1e-4, vf_lr=1e-4):
        # from spinup.algos.pytorch.Lappland.Isolated_primitive_trainer.RevampedWithAttentionDAPGForPrimitive import primitive_retrain
        goal =one_shot_demo['extracted_goal'][0]
        state_seq = one_shot_demo['state_same_dim']
        action_seq = one_shot_demo['action']

        one_shot_demo_processed = {'state':state_seq, 'action':action_seq, 'goal':goal}
        # primitive_retrain(primitive_policy=self.ac, one_shot_demo=one_shot_demo_processed, env=env,
        #                   pi_lr=pi_lr, id_lr=id_lr, vf_lr=vf_lr)
        print("this function is abandoned!")




    def rollout(self, env, goal, steps=0, render=False, get_normal_reward=False, consider_total_task_finish=False):
        """
        no_primitive_reward: if False, return the step_original reward
        :return:
        done means current primitive is finished, the total_task_finish signal is ret_info['total_task_done']
        """
        env.set_primitive_name(self.name)  # TODO: at present it depends on the environments
        env.set_primitive_goal(goal)

        ret_info = self.run(env, goal, steps=steps, render=render, get_normal_reward=get_normal_reward,
                            consider_total_task_finish=consider_total_task_finish)
        next_o, r, d, info = ret_info['current_terminal_state'], ret_info['ep_ret'], ret_info['done'], ret_info
        # info['total_task_done'] means total tasks done
        return next_o, r, d, info

    def evaluate_success(self, demo):
        self.env.reset()
        begin_state, end_state = demo['begin_env_state'], demo['end_env_state']
        state_seq, action_seq = demo['state'], demo['action']
        self.env.set_env_state(state_dict=begin_state)
        for action in action_seq:
            self.env.step(action)
            # self.env.render()
        # print(self.leave_condition(self.env))
        return self.leave_condition(self.env)

    def run(self, env, goal, steps=0, render=False, get_normal_reward=False, consider_total_task_finish=False):
        assert steps >= 0, 'steps should not be negative !'
        return_info = dict()
        '''
            2.7 added by quintus
            note that this input env should not necessarily be identical to self.env holden by current class
            but should be given after gym.make and env.reset()

            goal will be used as rl_goal as input

            when steps = 0 , the primitive will run till finish or meet its horizon
            else, the primitive will run as much as this step unless it terminate with steps less than provided

            render is a flag for visualization

            (important) the ac net should be loaded or trained first, for the action will be produced by this net
        '''

        def get_action(x, current_goal):
            ac = self.ac
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                # desired_state = model.act(x)
                # action = id(desired_state)
                # action = id(torch.cat((x, torch.as_tensor(desired_state))))
                action = ac.act(x, current_goal)
                # attention = ac.id.attention(x)
                # print(attention.numpy())
            return action

        if steps == 0:
            num_step = self.horizon
        else:
            num_step = steps

        primitive_ret = 0
        ep_len = 0
        o = env.get_obs()

        for t in range(num_step):

            if render:  # visualize
                env.render(extra_info = {'Goal Meaning': str(self.goal_meaning),
                                         'Goal Parameter':str(np.round(np.array(goal),2))})
                time.sleep(4e-2)

            rl_goal = goal  # TODO: check this rl_goal

            # rl_goal = torch.tensor(rl_goal)
            a = get_action(o, rl_goal)

            a = a.flatten()
            o, r, d, info = env.step(a)
            if get_normal_reward:
                r = env.get_origianl_step_reward()
            # print(r)
            done = info['primitives_goal_achieved'][self.primitive_id]
            if consider_total_task_finish:
                done = done or d
            # done = self.leave_condition(env=self.env, state=o, action=a)
            primitive_ret += r
            ep_len += 1

            if done or (ep_len == num_step - 1):
                if ep_len == num_step-1:
                    print("the primitive %s overtime" %(self.name))
                return_info['ep_ret'] = primitive_ret
                return_info['done'] = info['primitives_goal_achieved'][self.primitive_id]
                return_info['total_task_done'] = d
                return_info['current_terminal_state'] = o
                return_info['ep_len'] = ep_len
                return_info['timeout'] = False
                return_info['env_info'] = info
                if ep_len == num_step - 1:
                    return_info['timeout'] = True
                break

        return return_info

    def cal_state_loss(self, subtraj, goal):
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(subtraj)
        state = torch.as_tensor(subtraj[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(subtraj[1:state_len],dtype=torch.float32)
        goal = torch.as_tensor(goal,dtype=torch.float32).expand(state_len-1, len(goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()
        return error_sum





class ApproachPrimitive(Primitive):
    def __init__(self, env, ):
        super(ApproachPrimitive, self).__init__(env=env, name='Approach', goal_dim=3, observation_dim=39)
        self.primitive_id = 0
        self.horizon = 80
        self.goal_meaning = 'Ball Position'

    def __str__(self):
        return "Approach Primitive"

    def enter_condition(self, env, state=None, action=None):
        # full_state = env.get_env_state()
        # hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
        #     'target_pos'], full_state['palm_pos']
        # qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']
        # if np.linalg.norm(obj_pos - palm_pos) > 0.15:
        #     return True
        # else:
        #     return False
        return env.enter_condition(primitive_name=self.name)

    def leave_condition(self, env, state=None, action=None):
        # full_state = env.get_env_state()
        # hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
        #     'target_pos'], full_state['palm_pos']
        # qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']
        #
        # if np.linalg.norm(obj_pos - palm_pos) <= 0.08:
        #     return True
        # else:
        #     return False
        return env.leave_condition(primitive_name=self.name)

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(ApproachPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'obj_x':(-0.04, 0.04), 'obj_y':(-0.04, 0.04),
                             'obj_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, obj_x, obj_y, obj_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        primitive_pi = self.ac.pi
        error_sum = 0.0


        opt_goal = np.array([obj_x+self.inital_demo_goal[0],obj_y+self.inital_demo_goal[1],obj_z+self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        # s_temp = self.one_shot_demo_states[0]
        # gamma = 1.0 # discount factor
        # for i in range(len(self.one_shot_demo_states)-1): #1-length demontration is not allowed
        #     s_0 = np.concatenate((s_temp,opt_goal))
        #     # s_1 = np.concatenate((self.one_shot_demo_states[i+1],opt_goal))
        #     s_0 = torch.as_tensor(s_0, dtype=torch.float32).unsqueeze(dim=0)
        #
        #     s_1 = torch.as_tensor(self.one_shot_demo_states[i+1], dtype=torch.float32).unsqueeze(dim=0)
        #
        #     gamma *= 0.90
        #     if current_attention is not None:
        #         error_sum += gamma * ((primitive_pi(s_0)-s_1)**2*current_attention).sum().item()
        #     else:
        #         error_sum += gamma * (((primitive_pi(s_0)-s_1)**2).sum().item())
        #     s_temp = primitive_pi(s_0)[0].detach().numpy()
        # error_sum /= len(self.one_shot_demo_states)
        #---------------------------------------------------------------------------------------
        # for i in range(len(self.one_shot_demo_states)-1): #1-length demontration is not allowed
        #     s_0 = np.concatenate((self.one_shot_demo_states[i],opt_goal))
        #     # s_1 = np.concatenate((self.one_shot_demo_states[i+1],opt_goal))
        #     s_0 = torch.as_tensor(s_0, dtype=torch.float32).unsqueeze(dim=0)
        #     s_1 = torch.as_tensor(self.one_shot_demo_states[i+1], dtype=torch.float32).unsqueeze(dim=0)
        #
        #     if current_attention is not None:
        #         error_sum += ((primitive_pi(s_0)-s_1)**2*current_attention).mean().item()
        #     else:
        #         error_sum += ((primitive_pi(s_0)-s_1)**2).mean().item()
        # error_sum /= len(self.one_shot_demo_states)
        #---------------------------------------------------------------------------------------

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len],dtype=torch.float32)
        goal = torch.as_tensor(opt_goal,dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
         # goal is *obj_pos*
        return obs[-9:-6]

class GraspPrimitive(Primitive):
    def __init__(self, env):
        super(GraspPrimitive, self).__init__(env=env, name='Grasp', goal_dim=3, observation_dim=39)
        self.primitive_id = 1
        self.horizon = 80
        self.goal_meaning = 'Ball Position'

    def __str__(self):
        return "Grasp Primitive"

    def enter_condition(self, env, state=None, action=None):
        # full_state = env.get_env_state()
        # hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
        #     'target_pos'], full_state['palm_pos']
        # qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']
        # if obj_pos[2] < 0.04 and np.linalg.norm(palm_pos - obj_pos) < 0.10:
        #     return True
        # pass
        return env.enter_condition(primitive_name=self.name)

    def leave_condition(self, env, state=None, action=None):
        # full_state = env.get_env_state()
        # hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
        #     'target_pos'], full_state['palm_pos']
        # qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']
        #
        # # if np.linalg.norm(obj_pos - init_state['init_obj_pos']) >= 0.03 and np.linalg.norm(palm_pos - obj_pos) < 0.05:
        # #     return True

        # if obj_pos[2] >= 0.04 and np.linalg.norm(palm_pos - obj_pos) < 0.05:
        #     return True
        # else:
        #     return False
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *obj_pos*
        return obs[-9:-6]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(GraspPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'obj_x':(-0.04, 0.04), 'obj_y':(-0.04, 0.04),
                             'obj_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, obj_x, obj_y, obj_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([obj_x+self.inital_demo_goal[0],obj_y+self.inital_demo_goal[1],obj_z+self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len],dtype=torch.float32)
        goal = torch.as_tensor(opt_goal,dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum

class Move2targetPrimitive(Primitive):
    def __init__(self, env):
        super(Move2targetPrimitive, self).__init__(env=env, name='Move2Target', goal_dim=3, observation_dim=39)
        self.primitive_id = 2
        self.horizon = 80
        self.goal_meaning = 'Target Point Position'

    def __str__(self):
        return "Move-to-target Primitive"

    def enter_condition(self, env, state=None, action=None):
        # full_state = env.get_env_state()
        # hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
        #     'target_pos'], full_state['palm_pos']
        # qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']
        # if obj_pos[2] >= 0.04 and np.linalg.norm(palm_pos - obj_pos) < 0.05:
        #     return True
        # else:
        #     return False
        return env.enter_condition(primitive_name=self.name)

    def leave_condition(self, env, state=None, action=None):
        # full_state = env.get_env_state()
        # hand_qpos, obj_pos, target_pos, palm_pos = full_state['hand_qpos'], full_state['obj_pos'], full_state[
        #     'target_pos'], full_state['palm_pos']
        # qpos, qvel, init_state = full_state['qpos'], full_state['qvel'], full_state['init_state']
        #
        # # print("np.linalg.norm(obj_pos-init_state['init_target_obj_pos']): {}".format(np.linalg.norm(obj_pos-init_state['init_target_obj_pos'])))
        # # if np.linalg.norm(obj_pos - target_pos) <= 0.1 or np.linalg.norm(obj_pos - palm_pos) > 0.15:

        # if np.linalg.norm(obj_pos - target_pos) <= 0.04:
        #     return True
        # else:
        #     return False
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
         # goal is *target_pos*
        return obs[-3:]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(Move2targetPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'target_x':(-0.04, 0.04), 'target_y':(-0.04, 0.04),
                             'target_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, target_x, target_y, target_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([target_x+self.inital_demo_goal[0],target_y+self.inital_demo_goal[1],target_z+self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum


class DoorApproachPrimitive(Primitive):
    def __init__(self, env, ):
        super(DoorApproachPrimitive, self).__init__(env=env, name='DoorApproach')
        self.primitive_id = 0
        self.horizon = 80
        self.goal_meaning = 'Handle Position'

    def __str__(self):
        return "Door Approach Primitive"

    def enter_condition(self, env, state=None, action=None):
        return True

    def leave_condition(self, env, state=None, action=None):
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *handle_pos*
        return obs[-9:-6]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(DoorApproachPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'handle_x':(-0.04, 0.04), 'handle_y':(-0.04, 0.04),
                             'handle_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, handle_x, handle_y, handle_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([handle_x+self.inital_demo_goal[0],handle_y+self.inital_demo_goal[1],handle_z+self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum


class DoorGraspLatchPrimitive(Primitive):
    def __init__(self, env, ):
        super(DoorGraspLatchPrimitive, self).__init__(env=env, name='DoorGraspLatch')
        self.primitive_id = 1
        self.horizon = 80
        self.goal_meaning = 'Handle Position'

    def __str__(self):
        return "Door Grasp Latch Primitive"

    def enter_condition(self, env, state=None, action=None):
        return True

    def leave_condition(self, env, state=None, action=None):
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *handle_pos*
        return obs[-9:-6]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(DoorGraspLatchPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'handle_x':(-0.04, 0.04), 'handle_y':(-0.04, 0.04),
                             'handle_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, handle_x, handle_y, handle_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([handle_x+self.inital_demo_goal[0],handle_y+self.inital_demo_goal[1],handle_z+self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum


class DoorOpenPrimitive(Primitive):
    def __init__(self, env, ):
        super(DoorOpenPrimitive, self).__init__(env=env, name='DoorOpen')
        self.primitive_id = 2
        self.horizon = 80
        self.goal_meaning = 'Handle Position'

    def __str__(self):
        return "Door Approach Primitive"

    def enter_condition(self, env, state=None, action=None):
        return True

    def leave_condition(self, env, state=None, action=None):
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *handle_pos*
        return obs[-9:-6]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(DoorOpenPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'handle_x':(-0.04, 0.04), 'handle_y':(-0.04, 0.04),
                             'handle_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, handle_x, handle_y, handle_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([handle_x+self.inital_demo_goal[0],handle_y+self.inital_demo_goal[1],handle_z+self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum


class HammerApproachToolPrimitive(Primitive):
    def __init__(self, env, ):
        super(HammerApproachToolPrimitive, self).__init__(env=env, name='HammerApproachTool')
        self.primitive_id = 0
        self.horizon = 80
        self.goal_meaning = 'Hammer Position'

    def __str__(self):
        return "Hammer Approach Tool Primitive"

    def enter_condition(self, env, state=None, action=None):
        return True

    def leave_condition(self, env, state=None, action=None):
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *obj_pos*
        return obs[-9:-6]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(HammerApproachToolPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'obj_x':(-0.04, 0.04), 'obj_y':(-0.04, 0.04),
                             'obj_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, obj_x, ojb_y, obj_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([obj_x + self.inital_demo_goal[0], ojb_y + self.inital_demo_goal[1], obj_z + self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)

        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()

        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum



class HammerApproachNailPrimitive(Primitive): #TODO: the meaning is changed, not approach nail but grasp the tool
    def __init__(self, env, ):
        super(HammerApproachNailPrimitive, self).__init__(env=env, name='HammerApproachNail')
        self.primitive_id = 1
        self.init_state=dict()
        self.horizon = 80
        self.goal_meaning = 'Hammer Position'

    def __str__(self):
        return "Hammer Approach Nail Primitive"

    def get_env_init_info(self, env):
        """
        get the initial state to judge leave-condition
        """
        self.init_state = env.get_env_state()

    def enter_condition(self, env, state=None, action=None):
        return True

    def leave_condition(self, env, state=None, action=None):
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *obj_pos*
        return obs[-9:-6]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(HammerApproachNailPrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'obj_x':(-0.04, 0.04), 'obj_y':(-0.04, 0.04),
                             'obj_z':(-0.04, 0.04)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, obj_x, ojb_y, obj_z):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([obj_x + self.inital_demo_goal[0], ojb_y + self.inital_demo_goal[1], obj_z + self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)
        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()
        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum


class HammerNailGoInsidePrimitive(Primitive):
    def __init__(self, env, ):
        super(HammerNailGoInsidePrimitive, self).__init__(env=env, name='HammerNailGoInside')
        self.primitive_id = 2
        self.horizon = 80
        self.goal_meaning = 'Nail Position'

    def __str__(self):
        return "Hammer Nail Go Inside Primitive"

    def enter_condition(self, env, state=None, action=None):
        return True

    def leave_condition(self, env, state=None, action=None):
        return env.leave_condition(primitive_name=self.name)

    def extract_goal_from_obs_same_dim(self, obs): # extract goal from same-dim-obs
        # goal is *nail_pos_xy, [nail_impact]*
        return obs[-3:]

    def init_goal_optimization(self, one_shot_demonstration_states, inital_demo_goal):
        super(HammerNailGoInsidePrimitive, self).init_goal_optimization(one_shot_demonstration_states, inital_demo_goal)
        self.goal_pbounds = {'target_pos_x':(-0.04, 0.04), 'target_pos_y':(-0.04, 0.04),
                             'nail_impact':(-0.5, 0.5)} # TODO: the value should be adjusted with detecting accuracy
        self.inital_demo_goal = inital_demo_goal

    def goal_opt_function(self, target_pos_x, target_pos_y, nail_impact):
        # Note the obj_xxx is the incresament value
        assert not self.one_shot_demo_states==[], 'please input the subtraj demo first'
        opt_goal = np.array([target_pos_x + self.inital_demo_goal[0], target_pos_y + self.inital_demo_goal[1], nail_impact + self.inital_demo_goal[2]])
        current_attention = None
        if self.ac.id.attention_softmax:
            current_attention = self.ac.id.attention.clone().detach()
            current_attention = torch.nn.functional.softmax(current_attention, dim=0)
        state_len = len(self.one_shot_demo_states)
        state = torch.as_tensor(self.one_shot_demo_states[0:state_len-1], dtype=torch.float32)
        next_state = torch.as_tensor(self.one_shot_demo_states[1:state_len], dtype=torch.float32)
        goal = torch.as_tensor(opt_goal, dtype=torch.float32).expand(state_len-1, len(opt_goal))
        state_and_goal = torch.cat((state, goal), 1)
        expected_next_state = self.ac.pi(state_and_goal)
        error_sum = ((next_state - expected_next_state)**2).mean().detach().item()
        if current_attention is not None:
            error_sum = ((next_state - expected_next_state)**2*current_attention).mean().detach().item()
        error_sum = -error_sum # beyesian optimization  maxminizes the target-function
        return error_sum