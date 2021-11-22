# Data Collection for Lappland

This repository provides data collection for action primitives used in Lappland project.

The tasks we used to present our method are derived from [dapg](https://github.com/aravindr93/hand_dapg), including three dexterous hand manipulation tasks simulated in MuJoCo.

## Getting started

1. **Step 1:** Prepare your anaconda environment with ```python==3.6```

2. **Step 2:** Install [mujoco200](https://mujoco.org/), mujoco210 may cause some problems. (There is no need to install mujoco_py since we provide our own version in dependencies/we_mujoco_py)

3. **Step 3:** Activate your anaconda environment and run 

    ```
    $ pip install -r requirements.txt
    ```

4. **Step 4:** Install our own dependencies:

    ```
    $ cd dependencies
    $ pip install -e ./we_envs
    $ pip install -e ./we_mujoco
    $ pip install -e ./mjrl
    $ pip install -e ./gym-0.17.2
    ```
If you wish to see the environment and dexterous hand simulated in MuJoCo, you can find those ```.xml``` files in ```dependencies/we_envs/we_envs/we_robots/assets/mj_envs``` and you can drag them into Mujoco simulator for visualization.

## Collecting primitives

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

