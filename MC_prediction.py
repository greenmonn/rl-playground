import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from envs.gridworld import GridworldEnv
import plotting

# When we do not exactly know all about models

matplotlib.style.use('ggplot')

nx = 5
ny = 5

env = GridworldEnv([nx, ny])

class MCPrediction:
    def __init__(self):
        self.policy = np.ones([env.nS, env.nA]) / env.nA

    def estimate_value_function(self, env, num_episodes, discount_factor=1.0):
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        V = defaultdict(float)

        for i_episode in range(1, num_episodes + 1):
                if i_episode % 1000 == 0:
                    print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")

                episode = []
                state = env.reset()
                for t in range(100):
                    action = np.argmax(self.policy[state])
                    next_state, reward, done, _ = env.step(action)
                    episode.append((state, action, reward))
                    if done:
                        break
                    state = next_state

                states_in_episode = set([x[0] for x in episode])
                # remove duplicated states
                for state in states_in_episode:
                    first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)

                    G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])

                    returns_sum[state] += G
                    returns_count[state] += 1.0
                    state_tuple = (state % ny, state // ny)
                    V[state_tuple] = returns_sum[state] / returns_count[state]

        return V    




prediction = MCPrediction()

V_1k = prediction.estimate_value_function(env, num_episodes=1000)
plotting.plot_value_function(V_1k, title="1,000 Steps")

V_10k = prediction.estimate_value_function(env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")