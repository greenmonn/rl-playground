import numpy as np


class TDAgent:

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        self.gamma = gamma
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.epsilon = epsilon
        # self.n_step = n_step

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

        # Initialize "policy Q"
        # "policy Q" is the one used for policy generation.
        self._policy_q = None
        self.reset_policy()

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_policy(self):
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self.q[state, :].argmax()
        return action

    def update(self, episode):
        # perform 1-step TD (a.k.a TD(0))

        states, actions, rewards = episode
        next_states = states[1:] + [-1]  # -1 is a place holder
        dones = np.zeros_like(states)
        dones[-1] = 1

        for s, a, r, ns, done in zip(states, actions, rewards, next_states, dones):
            # TD target
            td_target = r + self.gamma * self.v[ns] * (1 - done)
            self.v[s] += self.lr * (td_target - self.v[s])

    def decaying_epsilon(self, factor):
        self.epsilon *= factor


class SARSA(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(SARSA, self).__init__(gamma=gamma,
                                    num_states=num_states,
                                    num_actions=num_actions,
                                    epsilon=epsilon,
                                    lr=lr)

    def reset_values(self):
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def update(self, episode):
        states, actions, rewards = episode
        next_states = states[1:] + [-1]  # -1 is a place holder
        dones = np.zeros_like(states)
        dones[-1] = 1

        for s, a, r, ns, done in zip(states, actions, rewards, next_states, dones):
            # SARSA target
            a_next = self.get_action(ns)
            td_target = r + self.gamma * self.q[ns, a_next] * (1 - done)
            self.q[s, a] += self.lr * (td_target - self.q[s, a])

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        a_next = self.get_action(ns)
        td_target = r + self.gamma * self.q[ns, a_next] * (1 - done)
        self.q[s, a] += self.lr * (td_target - self.q[s, a])


class QLeaner(TDAgent):

    def __init__(self,
                 gamma: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float,
                 lr: float):
        super(QLeaner, self).__init__(gamma=gamma,
                                      num_states=num_states,
                                      num_actions=num_actions,
                                      epsilon=epsilon,
                                      lr=lr)

    def reset_values(self):
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def update(self, episode):
        states, actions, rewards = episode
        next_states = states[1:] + [-1]  # -1 is a place holder
        dones = np.zeros_like(states)
        dones[-1] = 1

        for s, a, r, ns, done in zip(states, actions, rewards, next_states, dones):
            # Q-Learning target
            td_target = r + self.gamma * self.q[ns, :].max() * (1 - done)
            self.q[s, a] += self.lr * (td_target - self.q[s, a])

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        td_target = r + self.gamma * self.q[ns, :].max() * (1 - done)
        self.q[s, a] += self.lr * (td_target - self.q[s, a])
