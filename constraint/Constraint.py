import we_envs
import click
import os
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


class Constraint():
    def __init__(self, constraint_func=None, name=''):
        self.constraint_func = constraint_func
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, state):
        return self.constraint_func(state)

    def set_constraint(self, constraint_func):
        self.constraint_func = constraint_func


class ConstraintManager():
    def __init__(self):
        self.constraints = []

    def constraint_satisfied(self, state, name=''):
        if name == '':
            satisfied = True
            for constraint in self.constraints:
                if constraint(state) is False:
                    satisfied = False
            return satisfied
        related_constraint = self.find_constraint(name)
        return related_constraint(state)

    def add_constraint(self, constraint):
        assert isinstance(constraint, Constraint), "input is not the Constraint!"
        self.constraints.append(constraint)
        return

    def find_constraint(self, name=''):
        for i in range(len(self.constraints)):
            if self.constraints[i].name == name:
                return self.constraints[i]
        print("no related constraint found")
        return None

    def remove_constraint(self, name='', index=-1):
        assert not (index == -1 and name == ''), "please at least offer name or index"
        if index != -1:
            if index >= len(self.constraints):
                print("index over the limit")
                return False
            else:
                self.constraints.pop(index)
                return True

        elif name != '':
            removed = False
            for i in range(len(self.constraints)):
                if self.constraints[i].name == name:
                    self.constraints.pop(i)
                    removed = True
            return removed
