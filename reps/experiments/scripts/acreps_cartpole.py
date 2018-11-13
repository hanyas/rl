import numpy as np
from cluster_work import ClusterWork

from rl.reps.acreps_numpy import ACREPS

class MyExperiment(ClusterWork):

    _default_params = {
        'n_samples': 5000,
        'n_iter': 10,
        'n_rollouts': 25,
        'n_steps': 500,
        'n_keep': 0,
        'kl_bound': 0.1,
        'discount': 0.99,
        'trace': 0.95,
        'vreg': 1e-16,
        'preg': 1e-16,
        'cov0': 100.0,
        'n_vfeat': 250,
        'n_pfeat': 250,
        's_band': np.array([0.5, 0.5, 0.5, 12.5, 12.5]),
        'sa_band': np.array([0.5, 0.5, 0.5, 12.5, 12.5, 2.5]),
    }

    def reset(self, config=None, rep=0):
        n_samples = self._params['n_samples']
        n_iter = self._params['n_iter']
        n_rollouts = self._params['n_rollouts']
        n_steps = self._params['n_steps']
        n_keep = self._params['n_keep']
        kl_bound = self._params['kl_bound']
        discount = self._params['discount']
        trace = self._params['trace']
        vreg = self._params['vreg']
        preg = self._params['preg']
        cov0 = self._params['cov0']
        n_vfeat = self._params['n_vfeat']
        n_pfeat = self._params['n_pfeat']
        s_band = np.array(self._params['s_band'])
        sa_band = np.array(self._params['sa_band'])

        import gym
        import lab

        np.random.seed(self._seed)
        env = gym.make('Cartpole-v0')
        env._max_episode_steps = 500
        env.seed(self._seed)

        self.acreps = ACREPS(env=env,
                             n_samples=n_samples, n_iter=n_iter,
                             n_rollouts=n_rollouts, n_steps=n_steps, n_keep=n_keep,
                             kl_bound=kl_bound, discount=discount, trace=trace,
                             n_vfeat=n_vfeat, n_pfeat=n_pfeat,
                             vreg=vreg, preg=preg, cov0=cov0,
                             s_band=s_band, sa_band=sa_band)

    def iterate(self, config=None, rep=0, n=0):
        self.acreps.run(self.acreps.n_iter)
        return {}

if __name__ == "__main__":

    MyExperiment.run()
