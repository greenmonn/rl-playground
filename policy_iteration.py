import numpy as np
import pprint
import sys
from envs.gridworld import GridworldEnv


def evaluate_random_policy():
    env = GridworldEnv()
    DP.set_env(env)

    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = DP.policy_evaluation(random_policy)

    print("Value Function:")
    print(v.reshape(env.shape))
    print("")

def move_agent(env):
    # TODO: make Agent class
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    current_state = env.s  # start point

    print('Start Position: ({}, {})'.format(
        current_state // env.shape[1], current_state % env.shape[1]))

    env._render()

    done = False
    while not done:
        action = DP.get_action(current_state)
        observation, reward, done, info = env.step(action)

        current_state = observation
        print('Action: {}'.format(directions[action]))
        print('Reward: {}'.format(reward))
        env._render()

def move_by_value_iteration():
    env = GridworldEnv()
    DP.set_env(env)

    policy, V = DP.value_iteration()

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    move_agent(env)

def move_by_policy_iteration():
    env = GridworldEnv()
    DP.set_env(env)

    policy, V = DP.policy_iteration()

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    print("Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    move_agent(env)


class DP():
    discount_factor = 1.0
    theta = 0.00001

    @classmethod
    def set_env(cls, env):
        cls.env = env
        cls.policy = np.ones([env.nS, env.nA]) / env.nA

    @classmethod
    def policy_iteration(cls):
        policy_stable = False

        while not policy_stable:
            policy_stable, V = cls.policy_improvement()

        return cls.policy, V

    @classmethod
    def policy_evaluation(cls, policy=None):
        if policy == None:
            policy = cls.policy
        V = np.zeros(cls.env.nS)

        while True:
            delta = 0
            for s in range(cls.env.nS):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in cls.env.P[s][a]:
                        v += action_prob * prob * \
                            (reward + cls.discount_factor * V[next_state])

                delta = max(delta, np.abs(v - V[s]))
                V[s] = v

            if delta < cls.theta:
                break

        return np.array(V)

    @classmethod
    def policy_improvement(cls):
        V = cls.policy_evaluation()

        policy_stable = True

        for s in range(cls.env.nS):
            chosen_a = np.argmax(cls.policy[s])

            action_values = cls.one_step_lookahead(s, V)

            # Use list: breaking the tie
            max_indices = []
            max_value = -99999
            for i, value in enumerate(action_values):
                if value == max_value:
                    max_indices.append(i)
                elif value > max_value:
                    max_indices = [i]
                    max_value = value

            improved_policy = [0] * cls.env.nA
            prob = 1 / len(max_indices)
            for i in max_indices:
                improved_policy[i] = prob

            cls.policy[s] = improved_policy

            if chosen_a not in max_indices:
                policy_stable = False

        return policy_stable, V

    @classmethod
    def value_iteration(cls):
        V = np.zeros(cls.env.nS)
        while True:
            delta = 0

            for s in range(cls.env.nS):
                action_values = cls.one_step_lookahead(s, V)
                best_action_value = np.max(action_values)
                delta = max(delta, np.abs(best_action_value - V[s]))
                V[s] = best_action_value        
            if delta < cls.theta:
                break
        
        cls.policy = np.zeros([cls.env.nS, cls.env.nA])
        for s in range(cls.env.nS):
            action_values = cls.one_step_lookahead(s, V)
            # Use list: breaking the tie
            max_indices = []
            max_value = -99999
            for i, value in enumerate(action_values):
                if value == max_value:
                    max_indices.append(i)
                elif value > max_value:
                    max_indices = [i]
                    max_value = value

            improved_policy = [0] * cls.env.nA
            prob = 1 / len(max_indices)
            for i in max_indices:
                improved_policy[i] = prob

            cls.policy[s] = improved_policy
        
        return cls.policy, V

    @classmethod
    def get_action(cls, state):
        return np.argmax(cls.policy[state])

    @classmethod
    def one_step_lookahead(cls, state, V):
            A = np.zeros(cls.env.nA)
            for a in range(cls.env.nA):
                for prob, next_state, reward, done in cls.env.P[state][a]:
                    A[a] += prob * \
                        (reward + cls.discount_factor * V[next_state])
            return A


if __name__ == "__main__":
    move_by_value_iteration()
