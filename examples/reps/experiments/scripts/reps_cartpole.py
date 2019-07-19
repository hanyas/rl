import numpy as np
from cluster_work import ClusterWork

from rl.reps import REPS

class MyExperiment(ClusterWork):

    _default_params = {
        'nb_samples': 5000,
        'nb_keep': 0,
        'nb_rollouts': 25,
        'nb_steps': 500,
        'kl_bound': 0.1,
        'discount': 0.99,
        'vreg': 1e-16,
        'preg': 1e-16,
        'cov0': 100.0,
        'nb_vfeat': 250,
        'nb_pfeat': 250,
        'band': np.array([0.5, 0.5, 0.5, 12.5, 12.5]),
        'mult': 1.0
    }

    def reset(self, config=None, rep=0):
        nb_samples = self._params['nb_samples']
        nb_keep = self._params['nb_keep']
        nb_rollouts = self._params['nb_rollouts']
        nb_steps = self._params['nb_steps']
        kl_bound = self._params['kl_bound']
        discount = self._params['discount']
        vreg = self._params['vreg']
        preg = self._params['preg']
        cov0 = self._params['cov0']
        nb_vfeat = self._params['nb_vfeat']
        nb_pfeat = self._params['nb_pfeat']
        band = np.array(self._params['band'])
        mult = np.array(self._params['mult'])

        import gym

        np.random.seed(self._seed)
        env = gym.make('Cartpole-v1')
        env._max_episode_steps = 5000
        env.seed(self._seed)

        self.reps = REPS(env=env,
                         nb_samples=nb_samples, nb_keep=nb_keep,
                         nb_rollouts=nb_rollouts, nb_steps=nb_steps,
                         kl_bound=kl_bound, discount=discount,
                         vreg=vreg, preg=preg, cov0=cov0,
                         nb_vfeat=nb_vfeat, nb_pfeat=nb_pfeat,
                         band=band, mult=mult)

    def iterate(self, config=None, rep=0, n=0):
        return self.reps.run(nb_iter=1, verbose=True)


if __name__ == "__main__":
    MyExperiment.run()
