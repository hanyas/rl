import gym
from gym import envs
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')

n_states = env.observation_space.n
n_actions = env.action_space.n

v_function = np.ones((n_states,))

pi = 1 / n_actions * np.ones((n_actions,))

episodes = 1000
steps = 100

td_error = np.empty((0,))

gamma = 0.95
alpha = 0.1

for i in range(0, episodes):
	state = env.reset()
	action = np.argmax(np.random.multinomial(1, pi))

	for j in range(0, steps):
		next_state, reward, exit, _ = env.step(action)

		if not exit:
			v_function[state] += alpha * (reward + gamma * v_function[next_state] - v_function[state])
			td_error = np.append(td_error, reward + gamma * v_function[next_state] - v_function[state])
		if exit:
			v_function[state] += alpha * (reward - v_function[state])
			td_error = np.append(td_error, reward - v_function[state])

		state = next_state

print(v_function)
plt.plot(td_error)
plt.show()
