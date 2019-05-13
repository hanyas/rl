import numpy as np
from cluster_work import ClusterWork

from rl.reps.reps_npy import REPS

class MyExperiment(ClusterWork):

    _default_params = {
        'n_samples': 5000,
        'n_keep': 0,
        'n_rollouts': 25,
        'n_steps': 500,
        'kl_bound': 0.1,
        'discount': 0.98,
        'vreg': 1e-16,
        'preg': 1e-16,
        'cov0': 100.0,
        'n_vfeat': 250,
        'n_pfeat': 250,
        'band': np.array([1.57, 0.5, 0.5, 15.0, 20.0])
    }

    def reset(self, config=None, rep=0):
        n_samples = self._params['n_samples']
        n_keep = self._params['n_keep']
        n_rollouts = self._params['n_rollouts']
        n_steps = self._params['n_steps']
        kl_bound = self._params['kl_bound']
        discount = self._params['discount']
        vreg = self._params['vreg']
        preg = self._params['preg']
        cov0 = self._params['cov0']
        n_vfeat = self._params['n_vfeat']
        n_pfeat = self._params['n_pfeat']
        band = np.array(self._params['band'])

        import gym
        import lab

        np.random.seed(self._seed)
        env = gym.make('Qube-v1')
        env._max_episode_steps = 5000
        env.seed(self._seed)

        self.reps = REPS(env=env,
                         n_samples=n_samples, n_keep=n_keep,
                         n_rollouts=n_rollouts, n_steps=n_steps,
                         kl_bound=kl_bound, discount=discount,
                         vreg=vreg, preg=preg, cov0=cov0,
                         n_vfeat=n_vfeat, n_pfeat=n_pfeat,
                         band=band)

    def iterate(self, config=None, rep=0, n=0):
        rwrd, kls, kli, klm, ent = self.reps.run()

        print(f'rwrd={rwrd:{5}.{4}}',
              f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
              f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')
        return {}

if __name__ == "__main__":

    MyExperiment.run()