from gym.envs.registration import register
from gym.envs.registration import registry, register, make, spec


from we_envs.we_robots.ur5e.push import WeUR5ePushEnv
from we_envs.we_robots.ur5e.press import WeUR5ePressEnv

from we_envs.we_robots.ur5eShadowLite.mainpulate import WeUR5eShadowLiteEnv
from we_envs.we_robots.ur5eShadowLite.mainpulate_door import WeUR5eShadowLiteEnv_Door
from we_envs.we_robots.ur5eShadowLite.mainpulate_door_v2 import WeUR5eShadowLiteEnv_Door_V2

from we_envs.we_robots.AdroitHand.door_v0 import DoorEnvV0
from we_envs.we_robots.AdroitHand.hammer_v0 import HammerEnvV0
from we_envs.we_robots.AdroitHand.pen_v0 import PenEnvV0
from we_envs.we_robots.AdroitHand.relocate_v0 import RelocateEnvV0

from we_envs.we_robots.AdroitHand.relocate_v6 import RelocateEnvV6
from we_envs.we_robots.AdroitHand.door_v6 import DoorEnvV6
from we_envs.we_robots.AdroitHand.hammer_v6 import HammerEnvV6

from we_envs.we_robots.AdroitHand.readworlddoor_v0 import DoorEnvRealworldV0

from we_envs.we_robots.mini_cheetah.mini_cheetah_v0 import MiniCheetahEnvV0
from we_envs.we_robots.mini_cheetah.mini_cheetah_with_arm_v0 import MiniCheetahWithArmEnvV0

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='We_UR5ePush{}-v2'.format(suffix),
        entry_point='we_envs.we_robots:WeUR5ePushEnv',
        kwargs=kwargs,
        max_episode_steps=300,
    )

    register(
        id='We_UR5ePress{}-v2'.format(suffix),
        entry_point='we_envs.we_robots:WeUR5eCleanEnv',
        kwargs=kwargs,
        max_episode_steps=300,
    )


    ########################################
    register(
        id='We_UR5eShadowLite{}-v2'.format(suffix),
        entry_point='we_envs.we_robots:WeUR5eShadowLiteEnv',
        kwargs=kwargs,
        max_episode_steps=1000,
    )
    ########################################

register(
        id='We_UR5eShadowLiteDoor-v0',
        entry_point='we_envs.we_robots:WeUR5eShadowLiteEnv_Door',
        kwargs=kwargs,
        max_episode_steps=300,
    )

register(
        id='We_UR5eShadowLiteDoor-v2',
        entry_point='we_envs.we_robots:WeUR5eShadowLiteEnv_Door_V2',
        kwargs=kwargs,
        max_episode_steps=300,
    )


##################################################

register(
    id='Adroit-door-v0',
    entry_point='we_envs.we_robots:DoorEnvV0',
    max_episode_steps=200,
)

register(
    id='Adroit-hammer-v0',
    entry_point='we_envs.we_robots:HammerEnvV0',
    max_episode_steps=200,
)

register(
    id='Adroit-pen-v0',
    entry_point='we_envs.we_robots:PenEnvV0',
    max_episode_steps=200,
)

register(
    id='Adroit-relocate-v0',
    entry_point='we_envs.we_robots:RelocateEnvV0',
    max_episode_steps=200,
)

register(
    id='Adroit-relocate-v6',
    entry_point='we_envs.we_robots:RelocateEnvV6',
    max_episode_steps=240,
)


register(
    id='Adroit-door-v6',
    entry_point='we_envs.we_robots:DoorEnvV6',
    max_episode_steps=240,
)


register(
    id='Adroit-hammer-v6',
    entry_point='we_envs.we_robots:HammerEnvV6',
    max_episode_steps=240,
)

register(
    id='MiniCheetah-v0',
    entry_point='we_envs.we_robots:MiniCheetahEnvV0',
    max_episode_steps=2000,
)
register(
    id='MiniCheetah-withArm-v0',
    entry_point='we_envs.we_robots:MiniCheetahWithArmEnvV0',
    max_episode_steps=2000,
)

register(
    id='Realworld-door-v0',
    entry_point='we_envs.we_robots:DoorEnvRealworldV0',
    max_episode_steps=200,
)

#########################################