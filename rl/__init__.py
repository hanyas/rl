from gym.envs.registration import register


register(
    id='LQR-v0',
    entry_point='rl.envs:LQR',
    max_episode_steps=1000,
)

register(
    id='Hybrid-v0',
    entry_point='rl.envs:Hybrid',
    max_episode_steps=1000,
)

register(
    id='Lagoudakis-v0',
    entry_point='rl.envs:CartPole',
    max_episode_steps=1000,
)
