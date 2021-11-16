import gym
import we_envs
env = gym.make('We_UR5ePush-v2')

env = env.unwrapped

env.fix_env = True
env.reset()

while True:
    env.reset()
    for i in range(1000):
        action = [0.0,0.0,-0.01]
        obs, reward, done, info = env.step(action)
        # touchforce = env.getTouchforce()
        # print('touch_force:', touchforce)
        # fforce, tforce =  env.getFTforce()
        # print('ft_force:', fforce[:3])
        print(reward)
        env.render()
        # print(env.spec.timestep_limit)