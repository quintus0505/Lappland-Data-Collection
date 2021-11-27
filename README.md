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
    $ pip install -r requirements.txt
    ```
    - Note  we_mujoco is forked from [mujoco_py](https://github.com/openai/mujoco-py), mjrl is forked from [mjrl](https://github.com/aravindr93/mjrl).
    The original MJCF models and demonstrations are from [DAPG](https://github.com/aravindr93/hand_dapg).
    - MJCF models are available in ```dependencies/we_envs/we_envs/we_robots/assets/mj_envs```
4. **step 4:** visualize the task environments:
    The simulated environments are compatible with other [gym](https://github.com/openai/gym)'s environments. The following code demon
    ```python
    import gym, we_envs
    env = gym.make('Adroit-relocate-v6')
    obs = env.reset()
    while True:           
       action = env.action_space.sample()
       obs, reward, done, info = env.step(action)
       env.render()
    ```
    The available task envrionments are **Adroit-relocate-v6**, **Adroit-door-v6**, **Adroit-hammer-v6**.
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

