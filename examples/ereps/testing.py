from rl.ereps import eREPS
from rl.envs import Sphere


ereps = eREPS(func=Sphere(dm_act=5),
              n_episodes=10,
              kl_bound=0.1,
              cov0=10.0)

ereps.run(nb_iter=250, verbose=True)
