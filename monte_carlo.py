import numpy as np


class MCAgent:

    def __init__(self,
                 gamma: float,
                 lr: float,
                 num_states: int,
                 num_actions: int,
                 epsilon: float):
        self.gamma = gamma
        self.lr = lr
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon

        self._eps = 1e-5 # for stable computation of V and Q. NOT the one for e-greedy !

        # Initialize statistics
        self.n_v = None
        self.s_v = None
        self.n_q = None
        self.s_q = None
        self.reset_statistics()

        # Initialize state value function V and action value function Q
        self.v = None
        self.q = None
        self.reset_values()

    def reset_statistics(self):
        self.n_v = np.zeros(shape=self.num_states)
        self.s_v = np.zeros(shape=self.num_states)

        self.n_q = np.zeros(shape=(self.num_states, self.num_actions))
        self.s_q = np.zeros(shape=(self.num_states, self.num_actions))

    def reset_values(self):
        self.v = np.zeros(shape=self.num_states)
        self.q = np.zeros(shape=(self.num_states, self.num_actions))

    def update_stats(self, episode):
        states, actions, rewards = episode

        # reversing the inputs!
        # for efficient computation of returns
        states = reversed(states)
        actions = reversed(actions)
        rewards = reversed(rewards)

        iter = zip(states, actions, rewards)
        cum_r = 0
        for s, a, r in iter:
            cum_r *= self.gamma
            cum_r += r

            self.n_v[s] += 1
            self.n_q[s, a] += 1

            self.s_v[s] += cum_r
            self.s_q[s, a] += cum_r

    def compute_values(self):
        self.v = self.s_v / (self.n_v + self._eps)
        self.q = self.s_q / (self.n_q + self._eps)

    def get_action(self, state):
        prob = np.random.uniform(0.0, 1.0, 1)
        # e-greedy policy over Q
        if prob <= self.epsilon:  # random
            action = np.random.choice(range(self.num_actions))
        else:  # greedy
            action = self.q[state, :].argmax()
        return action

    def update_values(self, state):
        pass

    def improve_policy(self):
        self.reset_memory()

    def reset_policy(self):
        pass


def run_episode(env, agent):
    env.reset()
    states = []
    actions = []
    rewards = []

    while True:
        state = env.observe()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        if done:
            break

    episode = (states, actions, rewards)
    agent.update_stats(episode)


if __name__ == '__main__':
    from envs.gridworld import GridworldEnv

    nx, ny = 5, 5
    env = GridworldEnv([ny, nx])

    mc_agent = MCAgent(gamma=1.0,
                       lr=1e-3,
                       num_states=nx * ny,
                       num_actions=4,
                       epsilon=1.0)

    run_episode(env, mc_agent)
