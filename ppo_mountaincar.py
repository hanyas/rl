"""report bugs to riad@robot-learning.de"""
import tensorflow as tf
import numpy as np

import gym
import lab

import matplotlib.pyplot as plt

# 0: creating a feedforward neural network in tensorflow
class MLP:
    # sizes: [input length, hidden layer 1 length, ...,  hidden layer N length, output length]
    def __init__(self, sizes):
        self.x = last_out = tf.placeholder(dtype=tf.float32, shape=[None, sizes[0]])
        activations = [tf.tanh] * (len(sizes) - 2) + [tf.identity]
        # creating the feedforward neural net
        for size, activation in zip(sizes[1:], activations):
            # call tf.layers.dense(inputs, units (number of neurones), activation) to create a new layer
            # input of tf.layers.dense is a tensor (output of the previous layer)
            # output is another tensor
            # here last_out is used to store the last tensor
            # 0: --->

            last_out = tf.layers.dense(inputs=last_out, units=size, activation=activation)
        self.out = last_out


# 0: creating a Gaussian policy in tensorflow
class MLPGaussianPolicy:
    def __init__(self, act_dim, session, mlp, init_sigma=1.):
        self.mlp = mlp
        self.sess = session

        # action tensor (diagonal Gaussian)
        # logsigs: log of the std. deviation of the current policy
        # logsigs is a Variable (i.e. tensorflow will optimize wrt. logsigs
        # whenever it is asked to optimize a function parameterized by logsigs)
        self.logsigs = tf.Variable(np.log(init_sigma) * tf.ones([1, act_dim]))
        # create a tensor for a sampled action. Actions are sampled from a Gaussian of mean given by the NN
        # and std. given by exp(logsigs) (Variable above)
        # use tf.distributions.Normal(mean, stddev) to create a normal distribution
        # call function sample() from tf.distributions.Normal to sample an action
        # Reminder: class MLP stores output of the neural net in tensor 'out'
        # which you can access with mlp.out
        # 0: --->

        self.gauss_pol = tf.distributions.Normal(self.mlp.out, tf.exp(self.logsigs))
        self.act_tensor = self.gauss_pol.sample()
        
        # action proba (diagonal Gaussian)
        # tensor for the log density of the policy (needed for PPO's update)
        self.test_action = tf.placeholder(dtype=tf.float32, shape=[None, act_dim])
        self.log_prob = tf.reduce_sum(self.gauss_pol.log_prob(self.test_action), axis=1, keep_dims=True)

        # pol entropy (for logging only)
        self.entropy = tf.reduce_sum(self.logsigs + np.log(2 * np.pi * np.e) / 2)

    def get_action(self, obs):
        return np.squeeze(self.sess.run(self.act_tensor, {self.mlp.x: np.asmatrix(obs)}), axis=0)

    def get_log_proba(self, obs, act):
        return self.sess.run(self.log_prob, {self.mlp.x: obs, self.test_action: act})


# 1: learning the v function (policy evaluation)
class MLPVFunc:
    def __init__(self, session, mlp, lrate=1e-3):
        self.mlp = mlp
        self.sess = session
        
        # loss for v function
        self.target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # frame a regression problem minimizing squared error between
        # mlp(state) and target_v
        # use function tf.losses.mean_squared_error(labels, predictions)
        # 1: --->

        self.loss_v = tf.losses.mean_squared_error(labels=self.target_v, predictions=self.mlp.out)

        # call tf.train.AdamOptimizer(learning rate).minimize(a tensor "tens") to create a tensor that
        # computes the gradient of "tens" and does one Adam update of the Variables parameterizing "tens"
        # 1: --->

        self.optimizer_v = tf.train.AdamOptimizer(lrate).minimize(self.loss_v)
    
    def get_loss_v(self, obs, target_v):
        return self.sess.run(self.loss_v, {self.mlp.x: obs, self.target_v: target_v})

    def evaluate_pol(self, paths, discount=.999, lam_trace=.95, nb_epochs=50, batch_size=64):
        # estimate the v-function
        for epoch in range(nb_epochs):
            # compute the generalized td_error and v_targets
            v_values = self.sess.run(self.mlp.out, {self.mlp.x: paths["obs"]})
            gen_adv = np.empty_like(v_values)
            for rev_k, v in enumerate(reversed(v_values)):
                k = len(v_values) - rev_k - 1
                if paths["done"][k]:  # this is a new path. always true for rev_k == 0
                    gen_adv[k] = paths["rwd"][k] - v_values[k]
                else:
                    gen_adv[k] = paths["rwd"][k] + discount * v_values[k + 1] - v_values[k]  # TD(0)
                    # gen_adv[k] = paths["rwd"][k] + discount * v_values[k + 1] - v_values[k] + discount * lam_trace * gen_adv[k + 1]  # TD(lambda)
            v_targets = v_values + gen_adv  # generalized Bellmann operator

            # log Bellman error if first epoch
            if epoch == 0:
                print('v-function: loss before training is ', self.get_loss_v(paths["obs"], v_targets))
            for batch_idx in next_batch_idx(batch_size, len(v_targets)):
                # perform one gradient step with a mini-batch, to minimize mse between mlp output and v_targets
                # Reminder: we already created a tensor (i.e. a function) to perform this  (self.optimizer_v)
                # to 'call' a function in tensorflow, use session.run. We stored a default session in self.sess
                # you provide inputs to a function in tensorflow by passing a dictionary mapping placeholders to data
                # session.run(tensor to evaluate, {placeholder1: data, placeholder2: data, ...})
                # 1: --->

                self.sess.run(self.optimizer_v, {self.mlp.x: paths["obs"][batch_idx], self.target_v: v_targets[batch_idx]})

        print('v-function: loss after training is ', self.get_loss_v(paths["obs"], v_targets))
        return gen_adv


# 2: policy update (PPO)
class PPO:
    def __init__(self, session, policy, e_clip=.2, lrate=1e-4):
        self.sess = session
        self.pol = policy

        # clip loss for policy update
        # PPO's update depends on the advantage value of (s, a) samples
        # and on the log density of (s, a) of the data generating policy
        # create a tensorflow place holder for both the advantage and the log_probas
        # see MLPVFunc class for inspiration
        # 2: --->

        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.old_log_probas = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Now... PPO
        # write down PPO's loss. Useful functions: tf.exp, tf.clip_by_value(vals, min_val, max_val),
        # tf.reduce_mean(vals) to perform an average, tf.multiply for point-wise multiplication.
        # useful variables: self.pol.log_prob for the log density of an action conditioned on a state (see MLPGaussianPolicy for definition),
        # your placeholders for advantage and data generating log density and finally e_clip
        # 2: --->

        proba_ratio = tf.exp(self.pol.log_prob - self.old_log_probas)
        self.clip_pr = tf.clip_by_value(proba_ratio, 1 - e_clip, 1 + e_clip)
        self.neg_objective_act = -tf.reduce_mean(tf.minimum(tf.multiply(proba_ratio, self.advantage), tf.multiply(self.clip_pr, self.advantage)))

        # call tf.train.AdamOptimizer(learning rate).minimize(a tensor "tens") to create a tensor that
        # computes the gradient of "tens" and does one Adam update of the Variables parameterizing "tens"
        # 2: --->

        self.optimizer_act = tf.train.AdamOptimizer(lrate).minimize(self.neg_objective_act)

    def evaluate_pol(self, obs, old_act, old_log_probas, advantages):
        return -self.sess.run(self.neg_objective_act, {self.pol.mlp.x: obs, self.pol.test_action: old_act,
                                                       self.advantage: advantages, self.old_log_probas: old_log_probas})

    def update_pol(self, paths, adv, old_log_prob, nb_epochs=10, batch_size=64):
        print('entropy: before update', self.sess.run(self.pol.entropy))
        print('policy: objective before training ', self.evaluate_pol(paths["obs"], paths["act"], old_log_probas=old_log_prob, advantages=adv))
        for epoch in range(nb_epochs):
            for batch_idx in next_batch_idx(batch_size, len(adv)):
                # perform one gradient step with a mini-batch, to minimize PPO's loss
                # Reminder: we already created a tensor (i.e. a function) to perform this  (self.optimizer_act)
                # to 'call' a function in tensorflow, use session.run. Default session store in self.sess
                # you provide inputs to a function in tensorflow by passing a dictionary mapping placeholders to data
                # session.run(tensor to evaluate, {placeholder1: data, placeholder2: data, ...})
                # 2: --->

                self.sess.run(self.optimizer_act, {self.pol.mlp.x: paths["obs"][batch_idx], self.pol.test_action: paths["act"][batch_idx],
                                                     self.advantage: adv[batch_idx], self.old_log_probas: old_log_prob[batch_idx]})

        print('entropy: after update ', self.sess.run(self.pol.entropy))
        print('policy: objective after training ', self.evaluate_pol(paths["obs"], paths["act"], old_log_probas=old_log_prob, advantages=adv))


def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start:min(batch_start + batch_size, data_set_size)]


def rollout(env, policy, render=False):
    # Generates transitions until episode's end
    obs = env.reset()
    done = False
    while not done:
        if render:
            env.render()
        act = policy(obs)
        nobs, rwd, done, _ = env.step(np.minimum(np.maximum(act, env.action_space.low), env.action_space.high))
        yield obs, act, rwd, done
        obs = nobs


def rollouts(env, policy, min_trans, render=False):
    # Keep calling rollout and saving the resulting path until at least min_trans transitions are collected
    keys = ["obs", "act", "rwd", "done"]  # must match order of the yield above
    paths = {}
    for k in keys:
        paths[k] = []
    nb_paths = 0
    while len(paths["rwd"]) < min_trans:
        for trans_vect in rollout(env, policy, render):
            for key, val in zip(keys, trans_vect):
                paths[key].append(val)
        nb_paths += 1
    for key in keys:
        paths[key] = np.asarray(paths[key])
    paths["nb_paths"] = nb_paths
    return paths


# env = gym.make('MountainCarContinuous-v0')
env = gym.make('ContinuousMountainCar-v1')

# params
nb_iter = 25  # one iter -> at least min_trans_per_iter generated
min_trans_per_iter = 5000
exploration_sigma = 3.0
e_clip = .1  # the 'step size'

# mlp for v-function and policy
s_dim, a_dim = 2, 1

layer_sizes = [s_dim] + [16] * 2
pmlp = MLP(layer_sizes + [a_dim])  # 0
vmlp = MLP(layer_sizes + [1])  # 1

# policy, policy evaluation and policy update (PPO)
session = tf.Session()
policy = MLPGaussianPolicy(a_dim, session, pmlp, exploration_sigma)  # 0

vlearner = MLPVFunc(session, vmlp)  # 1
ppo = PPO(session, policy, e_clip=e_clip)  # 2

session.run(tf.global_variables_initializer())
for it in range(nb_iter):
    print('-------- iter', it, '--------')

    # Generates transition data by interacting with the env
    paths = rollouts(env, policy=policy.get_action, min_trans=min_trans_per_iter, render=False)  # 0
    print('avg_rwd', np.sum(paths["rwd"]) / paths["nb_paths"]) # average (undiscounted) reward  # 0

    # policy eval
    adv = vlearner.evaluate_pol(paths)  # 1

    # policy update
    log_act_probas = policy.get_log_proba(paths["obs"], paths["act"])  # 2
    ppo.update_pol(paths, adv, log_act_probas)  # 2
