# Data Collection for Lappland

This repository provides demonstration collection and simulated environments for Lappland (layered action primitive planning  from  demonstration) project.

## Getting started

1. **Step 1:** Prepare your  environment with ```ubuntu>=16.04``` , ```python==3.6```

2. **Step 2:** Install [mujoco200](https://mujoco.org/) (mujoco210 may cause some problems). 

3. **Step 3:** Install following extra dependencies:

    ```
    $ cd dependencies
    $ pip install -e ./gym-0.17.2
    $ pip install -e ./we_mujoco
    $ pip install -e ./mjrl
    $ pip install -e ./we_envs
    ```
    Note  we_mujoco is forked from [mujoco_py](https://github.com/openai/mujoco-py), mjrl is forked from [mjrl](https://github.com/aravindr93/mjrl).
    The original MJCF models and demonstrations are from [DAPG](https://github.com/aravindr93/hand_dapg) 
4. **Step 4:** Install following  dependencies: 

    ```
    $ pip install -r requirements.txt
    ```

If you wish to see the environment and dexterous hand simulated in MuJoCo, you can find those ```.xml``` files in ```dependencies/we_envs/we_envs/we_robots/assets/mj_envs``` and you can drag them into Mujoco simulator for visualization.

## Collecting primitive demonstrations

For **Relocate** task, you can run the following command:

```
$ python collect_primitives/auto_collect_relocate_primitive.py --option collect
```
We also provide other two tasks, the primitives of which can be collected by similar instruction.

## Visualizing primitives

The collected primitives are saved as pickle files in ```collect_primitives/```. For visualization, please run the following code:

```
$ python collect_primitives/auto_collect_relocate_primitive.py --option visualize --primitive_name Approach
```

The tasks we provide all have three primitives, you can find their names by running the following command:

```
$ python collect_primitives/auto_collect_relocate_primitive.py --help
```

