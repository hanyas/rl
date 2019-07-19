import os
import pickle

from gym.envs.registration import register


register(
    id='LQR-RL-v0',
    entry_point='rl.envs:LQR',
    max_episode_steps=1000,
)

register(
    id='Pendulum-RL-v0',
    entry_point='rl.envs:Pendulum',
    max_episode_steps=1000,
)

register(
    id='MassSpringDamper-RL-v0',
    entry_point='rl.envs:MassSpringDamper',
    max_episode_steps=1000,
)

register(
    id='Lagoudakis-RL-v0',
    entry_point='rl.envs:CartPole',
    max_episode_steps=1000,
)

try:
    register(
        id='HybridMassSpringDamper-RL-v0',
        entry_point='rl.envs:HybridMassSpringDamper',
        max_episode_steps=1000,
        kwargs={'rarhmm': pickle.load(open(os.path.dirname(__file__)
                                           + '/envs/control/hybrid/models/hybrid_msd.p', 'rb'))}
    )
except:
    pass

try:
    register(
        id='HybridPendulum-RL-v0',
        entry_point='rl.envs:HybridPendulum',
        max_episode_steps=1000,
        kwargs={'rarhmm': pickle.load(open(os.path.dirname(__file__)
                                           + '/envs/control/hybrid/models/hybrid_pendulum.p', 'rb'))}
    )
except:
    pass
