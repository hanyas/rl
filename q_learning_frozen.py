import gym
from gym import envs
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')


def greedy(state, eps, n_actions):
	pmf = eps / n_actions * np.ones((n_actions, 1))
	idx = np.argmax(q_function[state])
	pmf[idx] = 1.0 - np.sum(np.concatenate((pmf[:idx], pmf[(idx + 1):])), axis=0)
	p = np.random.multinomial(1, pmf)
	return np.argmax(p)


def softmax(state, q_function, beta):
	pmf = np.exp(q_function[state] / beta) / np.sum(np.exp(q_function[state] / beta))
	p = np.random.multinomial(1, pmf)
	return np.argmax(p)


n_states = env.observation_space.n
n_actions = env.action_space.n

q_function = 10 * np.ones((n_states, n_actions))

score = np.empty((0, 1))
td_error = np.empty((0, 1))

episodes = 1000
steps = 500

alpha = 0.2
gamma = 0.99
beta = 0.98

for i in range(0, episodes):
	state = env.reset()
	# eps = 1.0 / (1.0 * (i + 1.0))
	beta = beta * 0.995

	for j in range(0, steps):
		# action = greedy(state, eps, n_actions)
		action = softmax(state, q_function, beta)
		next_state, reward, exit, _ = env.step(action)

		if not exit:
			q_function[state, action] += alpha * (reward + gamma * np.max(q_function[next_state, :]) - q_function[state, action])
			td_error = np.append(td_error, reward + gamma * np.max(q_function[next_state, :]) - q_function[state, action])
		if exit:
			q_function[state, action] += alpha * (reward - q_function[state, action])
			td_error = np.append(td_error, reward - q_function[state, action])

		state = next_state

		if exit:
			if len(score) < 100:
				score = np.append(score, reward)
			else:
				score[i % 100] = reward

			print("Episode: {} Step: {} R={} Score: {}".format(i, j, reward, np.mean(score, axis=0)))
			break

print(q_function)
plt.plot(td_error)
plt.show()
