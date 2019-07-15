from gym.envs.registration import register


register(
    id='LQR-v0',
    entry_point='rl.envs.control.lqr.lqr:LQR',
    max_episode_steps=1000,
)

register(
    id='Lagoudakis-v0',
    entry_point='rl.envs.control.cartpole.cartpole:CartPole',
    max_episode_steps=1000,
)
