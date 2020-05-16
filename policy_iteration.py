import numpy as np
import pprint
import sys
from envs.gridworld import GridworldEnv


def evaluate_random_policy():
    env = GridworldEnv()
    PolicyIteration.set_env(env)

    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = PolicyIteration.policy_evaluation(random_policy)

    print("Value Function:")
    print(v.reshape(env.shape))
    print("")


def improve_policy_until_stable():
    env = GridworldEnv()
    PolicyIteration.set_env(env)

    policy_stable = False

    while not policy_stable:
        policy_stable = PolicyIteration.policy_improvement()

    policy = PolicyIteration.policy

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")


def move_by_best_policy():
    env = GridworldEnv()
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    PolicyIteration.set_env(env)

    PolicyIteration.run()

    current_state = env.s  # start point

    print('Start Position: ({}, {})'.format(
        current_state // env.shape[1], current_state % env.shape[1]))

    print('===========')
    env._render()
    print('===========')

    done = False
    while not done:
        action = PolicyIteration.get_action(current_state)
        observation, reward, done, info = env.step(action)

        current_state = observation
        print('Action: {}'.format(directions[action]))
        print('===========')
        env._render()
        print('===========')


class PolicyIteration():
    discount_factor = 1.0
    theta = 0.00001

    @classmethod
    def set_env(cls, env):
        cls.env = env
        cls.policy = np.ones([env.nS, env.nA]) / env.nA

    @classmethod
    def run(cls):
        policy_stable = False

        while not policy_stable:
            policy_stable = cls.policy_improvement()

        return cls.policy

    @classmethod
    def policy_evaluation(cls):
        V = np.zeros(cls.env.nS)

        while True:
            delta = 0
            for s in range(cls.env.nS):
                v = 0
                for a, action_prob in enumerate(cls.policy[s]):
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
        def one_step_lookahead(state, V):
            A = np.zeros(cls.env.nA)
            for a in range(cls.env.nA):
                for prob, next_state, reward, done in cls.env.P[state][a]:
                    A[a] += prob * \
                        (reward + cls.discount_factor * V[next_state])
            return A

        V = cls.policy_evaluation()

        policy_stable = True

        for s in range(cls.env.nS):
            chosen_a = np.argmax(cls.policy[s])

            action_values = one_step_lookahead(s, V)

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

            best_a = np.argmax(action_values)

            if chosen_a not in max_indices:
                policy_stable = False

        return policy_stable

    @classmethod
    def get_action(cls, state):
        return np.argmax(cls.policy[state])


if __name__ == "__main__":
    move_by_best_policy()
