import gym
from gym import envs
import numpy as np
from matplotlib import pyplot as plt


def greedy(state, eps, n_actions):
	pmf = eps / n_actions * np.ones((n_actions,))
	idx = np.argmax(q_function[state])
	pmf[idx] = 1.0 - np.sum(np.concatenate((pmf[:idx], pmf[(idx + 1):])), axis=0)
	p = np.random.multinomial(1, pmf)
	return np.argmax(p)


def softmax(state, q_function, beta):
	pmf = np.exp(q_function[state] / beta) / np.sum(np.exp(q_function[state] / beta))
	p = np.random.multinomial(1, pmf)
	return np.argmax(p)


env = gym.make('FrozenLake-v0')

n_states = env.observation_space.n
n_actions = env.action_space.n

q_function = 10 * np.ones((n_states, n_actions))

score = np.empty((0,))
td_error = np.empty((0,))

episodes = 10000
steps = 500

alpha = 0.2
gamma = 0.99
beta = 0.98

data_store = np.zeros((episodes, steps + 1, 3))

for i in range(0, episodes):
	state = env.reset()
	# eps = 1.0 / (1.0 * (i + 1.0))
	# action = greedy(state, eps, n_actions)
	beta = beta * 0.9995
	action = softmax(state, q_function, beta)

	data_store[i, 0, 0] = state
	data_store[i, 0, 1] = action

	for j in range(0, steps):
		next_state, reward, exit, _ = env.step(action)
		# next_action = greedy(next_state, eps, n_actions)
		next_action = softmax(next_state, q_function, beta)

		if not exit:
			q_function[state, action] += alpha * (reward + gamma * q_function[next_state, next_action] - q_function[state, action])
			td_error = np.append(td_error, reward + gamma * q_function[next_state, next_action] - q_function[state, action])
		if exit:
			q_function[state, action] += alpha * (reward - q_function[state, action])
			td_error = np.append(td_error, reward - q_function[state, action])

		state = next_state
		action = next_action

		data_store[i, j, 2] = next_state
		data_store[i, j + 1, 0] = state
		data_store[i, j + 1, 1] = action

		if exit:
			data_store[i, j + 1:, 2] = -99
			data_store[i, j + 1 + 1:, 0] = -99
			data_store[i, j + 1 + 1:, 1] = -99

			if len(score) < 100:
				score = np.append(score, reward)
			else:
				score[i % 100] = reward

			print("Episode: {} Step: {} R={} Score: {}".format(i, j, reward, np.mean(score, axis=0)))
			break

print(q_function)
plt.plot(td_error)
plt.show()

ds = data_store[5000:, :, :].reshape((-1, 3))
ds = ds[ds[:, 0] != -99]
ds = ds[ds[:, -1] != -99]

np.savetxt("data/frozen_lake.csv", ds, fmt='%i', delimiter=",")