from sandbox.rocky.tf.envs.base import TfEnv
from inverse_rl.envs.env_utils import CustomGymEnv
import gym

env = TfEnv(CustomGymEnv('CustomAnt-v0', record_video=False, record_log=False, force_reset=False))
done = False
obs = env.reset()
while not done:
    env.render()
    action = env.action_space.sample()
    _,_,done,_= env.step(action)

env = TfEnv(CustomGymEnv('DisabledAnt-v0', record_video=False, record_log=False, force_reset=False))
done = False
obs = env.reset()
while not done:
    env.render()
    action = env.action_space.sample()
    _,_,done,_= env.step(action)