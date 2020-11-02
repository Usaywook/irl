import sys
sys.path.append('/media/usaywook/rllab')
import tensorflow as tf

from inverse_rl.algos.trpo import TRPO
from inverse_rl.models.tf_util import get_session_config, load_prior_params
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial
import pdb
import numpy as np
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.sampler.utils import rollout

# env = TfEnv(CustomGymEnv('CustomAnt-v0', record_video=False, record_log=False, force_reset=False))
env = TfEnv(CustomGymEnv('DisabledAnt-v0', record_video=False, record_log=False, force_reset=False))

# logdir = '/home/usaywook/ext256/inverse_rl/data/ant_state_irl/itr_2999.pkl'
logdir = '/home/usaywook/ext256/inverse_rl/data/ant_transfer/itr_1500.pkl'
params = load_prior_params(logdir)
loaded_params = params['policy_params']

policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if loaded_params is not None:
        # x = list(params['policy']._cached_params.values())[0]
        # y = list(params['policy']._cached_param_dtypes.values())[0]
        policy.set_param_values(loaded_params)
# pdb.set_trace()

with tf.Session(config=get_session_config()) as sess:
    # algo = TRPO(
    #     env=env,
    #     sess=sess,
    #     policy=policy,
    #     n_itr=1,
    #     batch_size=20000,
    #     max_path_length=500,
    #     discount=0.99,
    #     store_paths=True,
    #     entropy_weight=0.1,
    #     baseline=LinearFeatureBaseline(env_spec=env.spec),
    #     exp_name=None,
    #     plot=True
    # )
    # algo.train()
    # if policy.vectorized:
    #     sampler_cls = VectorizedSampler
    # else:
    #     sampler_cls = BatchSampler
    created_session = True if (sess is None) else False
    if sess is None:
        sess = tf.Session()
        sess.__enter__()
    n_itr = 1
    sess.run(tf.global_variables_initializer())
    # sampler_cls.start_worker()
    for itr in range(n_itr):
        # rollout(env, policy, animated=True, max_path_length=1000)
        o = env.reset()
        policy.reset()
        d = False
        while not d:
            env.render()
            flat_obs = policy.observation_space.flatten(o)
            mean, log_std = [x[0] for x in policy._f_dist([flat_obs])]
            # rnd = np.random.normal(size=mean.shape)
            # action = rnd * np.exp(log_std) + mean
            action = mean
            next_o, r, d, env_info = env.step(action)
            o = next_o

    # sampler_cls.shutdown_worker()
    if created_session:
        sess.close()

    # done = False
    # obs = env.reset()
    # rewards = []
    # pdb.set_trace()
    # while not done:
    #     action, actor_info = policy.get_actions(obs.reshape(-1,111))
    #     obs, reward, done, info =  env.step(action)
    #     rewards.append(reward)
    #
    # pdb.set_trace()