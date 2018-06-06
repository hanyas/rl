import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1337)

# reward
R = np.array([1.0, 0.0])

# transition
# P(k,i,j) = p(s' = j | s = i, a = k)
P = np.array([[[0.95, 0.05], [0.05, 0.95]],
              [[0.95, 0.05], [0.05, 0.95]]])

v_function = np.zeros((2, ))

pi = np.array([[0.99, 0.01], [0.01, 0.99]])

episodes = 1000
steps = 25

td_error = np.empty((0,))

gamma = 0.92
alpha = 0.9

for i in range(episodes):
	state = 0

	for j in range(steps):
		action = np.argmax(np.random.multinomial(1, pi[state, :]))

		p = P[action, state, :]
		next_state = np.argmax(np.random.multinomial(1, p))

		td_error = np.append(td_error, R[state] + gamma * v_function[next_state] - v_function[state])
		v_function[state] += alpha * td_error[-1]

		state = next_state.copy()

print(v_function)
plt.plot(td_error)
plt.show()
