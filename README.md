# Data Collection for Lappland

This repository provides data collection for action primitives used in Lappland project.

## Getting started

1. **Step 1:** Prepare your conda environment with ```python==3.6```

2. **Step 2:** Install [mujoco200](https://mujoco.org/), mujoco210 may cause some problems. (There is no need to install mujoco_py since we provide our own version in dependencies/we_mujoco_py)

3. **Step 3:** Activate your conda environment and run 

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
## Running the code
```
$ python collect_primitives/auto_collect_relocate_primitive.py --option collect

$ python collect_primitives/auto_collect_relocate_primitive.py --option visualize --primitive_name Approach
```
