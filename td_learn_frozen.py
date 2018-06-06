import gym
from gym import envs
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1337)

env = gym.make('FrozenLake-v0')

n_states = env.observation_space.n
n_actions = env.action_space.n

v_function = np.zeros((n_states,))

pi = 1.0 / n_actions * np.ones((n_actions,))

episodes = 1000
steps = 50

td_error = np.empty((0,))

gamma = 0.95
alpha = 0.9

for i in range(episodes):
	state = env.reset()

	for j in range(steps):
		action = np.argmax(np.random.multinomial(1, pi))
		next_state, reward, exit, _ = env.step(action)

		if not exit:
			v_function[state] += alpha * (reward + gamma * v_function[next_state] - v_function[state])
			td_error = np.append(td_error, reward + gamma * v_function[next_state] - v_function[state])
		if exit:
			v_function[state] += alpha * (reward - v_function[state])
			td_error = np.append(td_error, reward - v_function[state])

		state = next_state

v_function = v_function / v_function.sum()

print(v_function)
plt.plot(td_error)
plt.show()
