
import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os


model = load_model_from_path("reach.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 1
    sim.data.ctrl[1] = math.cos(t / 10.) * 1
    # t += 1
    for i in range(10):
        sim.step()
        viewer.render()
