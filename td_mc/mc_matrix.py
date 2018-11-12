import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1337)

# reward
R = np.array([1.0, 0.0])

# transition
# P(k,i,j) = p(s' = j | s = i, a = k)
P = np.array([[[0.95, 0.05], [0.05, 0.95]],
              [[0.95, 0.05], [0.05, 0.95]]])

v_function = np.zeros((2,))

pi = np.array([[0.99, 0.01], [0.01, 0.99]])

episodes = 1000
steps = 25

returns = np.zeros((episodes, steps))
states = np.zeros((episodes, steps), np.int64)

for i in range(episodes):
	states[i, 0] = 0

	for j in range(1, steps):
		returns[i, j - 1] = R[states[i, j - 1]]
		action = np.argmax(np.random.multinomial(1, pi[states[i, j - 1], :]))

		p = P[action, states[i, j - 1], :]
		next_state = np.argmax(np.random.multinomial(1, p))

		states[i, j] = next_state

for i in range(episodes):
	for j in range(2):
		idx = np.where(states[i, :] == j)
		if idx[0].size==0:
			continue
		else:
			idx = np.amin(idx)
			v_function[j] += returns[i, idx:].sum()

v_function = v_function / episodes
print(v_function)