import numpy as np
from matplotlib import pyplot as plt

# reward
R = np.array([1.0, 0.0])

# transition
# P(k,i,j) = p(s' = j | s = i, a = k)
P = np.array([[[1.0, 0.0], [1.0, 0.0]],
              [[0.0, 1.0], [0.0, 1.0]]])

v_function = np.zeros((2,))

pi = 0.5 * np.ones((2,))

episodes = 1000
td_error = np.empty((0,))

gamma = 0.98
alpha = 0.5

state = 0

for i in range(0, episodes):
	action = np.argmax(np.random.multinomial(1, pi))

	p = P[action, state, :]
	print(p)
	next_state = np.argmax(np.random.multinomial(1, p))

	td_error = np.append(td_error, R[state] + gamma * v_function[next_state] - v_function[state])
	v_function[state] = v_function[state] + alpha * td_error[i]

	state = next_state

print(v_function)
plt.plot(td_error)
plt.show()
