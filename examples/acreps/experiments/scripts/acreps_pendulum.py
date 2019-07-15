import numpy as np
from cluster_work import ClusterWork

from rl.rl.acreps import ACREPS

class MyExperiment(ClusterWork):

    _default_params = {
        'n_samples': 5000,
        'n_keep': 0,
        'n_rollouts': 25,
        'kl_bound': 0.1,
        'discount': 0.98,
        'lmbda': 0.95,
        'vreg': 1e-16,
        'preg': 1e-12,
        'cov0': 16.0,
        'n_vfeat': 75,
        'n_pfeat': 75,
        's_band': np.array([0.5, 0.5, 4.0]),
        'sa_band': np.array([0.5, 0.5, 4.0, 1.0]),
    }

    def reset(self, config=None, rep=0):
        n_samples = self._params['n_samples']
        n_keep = self._params['n_keep']
        n_rollouts = self._params['n_rollouts']
        kl_bound = self._params['kl_bound']
        discount = self._params['discount']
        lmbda = self._params['lmbda']
        vreg = self._params['vreg']
        preg = self._params['preg']
        cov0 = self._params['cov0']
        n_vfeat = self._params['n_vfeat']
        n_pfeat = self._params['n_pfeat']
        s_band = np.array(self._params['s_band'])
        sa_band = np.array(self._params['sa_band'])

        import gym

        np.random.seed(self._seed)
        env = gym.make('Pendulum-v0')
        env._max_episode_steps = 200
        env.seed(self._seed)

        self.acreps = ACREPS(env=env,
                             n_samples=n_samples, n_keep=n_keep, n_rollouts=n_rollouts,
                             kl_bound=kl_bound, discount=discount, lmbda=lmbda,
                             vreg=vreg, preg=preg, cov0=cov0,
                             n_vfeat=n_vfeat, n_pfeat=n_pfeat,
                             s_band=s_band, sa_band=sa_band)

    def iterate(self, config=None, rep=0, n=0):
        self.acreps.run(nb_iter=1, verbose=True)
        return {}


if __name__ == "__main__":
    MyExperiment.run()
