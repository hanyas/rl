import os
import torch

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
    id='Pendulum-RL-v1',
    entry_point='rl.envs:PendulumWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='MassSpringDamper-RL-v0',
    entry_point='rl.envs:MassSpringDamper',
    max_episode_steps=1000,
)

register(
    id='Cartpole-RL-v0',
    entry_point='rl.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='Cartpole-RL-v1',
    entry_point='rl.envs:CartpoleWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='LagoudakisCartpole-RL-v0',
    entry_point='rl.envs:LagoudakisCartpole',
    max_episode_steps=1000,
)

register(
    id='QQube-RL-v0',
    entry_point='rl.envs:QQube',
    max_episode_steps=1000,
    kwargs={'fs': 500.0, 'fs_ctrl': 100.0}
)

try:
    register(
        id='HybridMassSpringDamper-RL-v0',
        entry_point='rl.envs:HybridMassSpringDamper',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/control/hybrid/models/poly_rarhmm_msd.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass

try:
    register(
        id='HybridPendulum-RL-v0',
        entry_point='rl.envs:HybridPendulum',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/control/hybrid/models/neural_rarhmm_pendulum_polar.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass

try:
    register(
        id='HybridPendulum-RL-v1',
        entry_point='rl.envs:HybridPendulumWithCartesianObservation',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/control/hybrid/models/neural_rarhmm_pendulum_cart.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass
