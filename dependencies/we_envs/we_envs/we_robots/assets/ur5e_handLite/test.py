
import numpy as np

from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os


model = load_model_from_path("manipulate_egg.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    # sim.data.ctrl[0] = 3.14+1.57
    sim.data.ctrl[0] = -1.57
    sim.data.ctrl[6] = -0.03
    # sim.data.ctrl[2] = 0
    # sim.data.ctrl[3] = 0
    # sim.data.ctrl[4] = 0
    # sim.data.ctrl[5] = 0
    # sim.data.ctrl[6] = 0
    # sim.data.ctrl[6] = t
    t += 1
    for i in range(100):
        sim.step()
        viewer.render()
    # print(sim.data.sensordata[6])
