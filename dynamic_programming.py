import numpy as np
import pprint
import sys
from envs.gridworld import GridworldEnv

from queue import PriorityQueue

# TODO (discuss) reward: receive 0.0 when terminal state, but -1.0 when just entering the terminal state
# should fix?

# Activity: how to break ties?


class DP():
    def __init__(self, is_synchronous=False, is_prioritised=False, env=None):
        self.discount_factor = 1.0
        self.theta = 0.00001
        self.delta_list = []    # to check convergence
        self.is_synchronous = is_synchronous
        self.is_prioritised = is_prioritised

        if env != None:
            self.env = env
            self.policy = np.ones([env.nS, env.nA]) / env.nA

    def set_env(self, env):
        self.env = env
        self.policy = np.ones([env.nS, env.nA]) / env.nA

    def policy_iteration(self):
        policy_stable = False

        while not policy_stable:
            policy_stable, V = self.policy_improvement()

        return self.policy, V

    def policy_evaluation(self, policy=None):
        if policy == None:
            policy = self.policy
        V = np.zeros(self.env.nS)

        while True:
            max_delta = 0
            for s in range(self.env.nS):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    # Not known in RL
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v += action_prob * prob * \
                            (reward + self.discount_factor * V[next_state])

                max_delta = max(max_delta, np.abs(v - V[s]))
                V[s] = v

            if max_delta < self.theta:
                break

        return np.array(V)

    def policy_improvement(self):
        V = self.policy_evaluation()

        policy_stable = True

        for s in range(self.env.nS):
            chosen_action = np.argmax(self.policy[s])

            action_values = self.one_step_lookahead(s, V)
            best_action = np.argmax(action_values)

            self.policy[s] = np.eye(self.env.nA)[best_action]

            if chosen_action != best_action:
                policy_stable = False

        return policy_stable, V

    def _update_value_prioritised(self, V):
        bellman_errors = PriorityQueue()
        # initialize queue
        for s in range(self.env.nS):
            bellman_errors.put((0, s))

        while True:
            max_delta = 0
            sum_delta = 0

            bellman_errors_next = PriorityQueue()

            while not bellman_errors.empty():
                s = bellman_errors.get()[1]

                action_values = self.one_step_lookahead(s, V)
                best_action_value = np.max(action_values)
                
                error = abs(best_action_value - V[s])
                bellman_errors_next.put((-error, s))
                max_delta = max(max_delta, error)
                sum_delta += error

                V[s] = best_action_value
            
            bellman_errors = bellman_errors_next
            
            self.delta_list.append(sum_delta / self.env.nS)
            if max_delta < self.theta:
                break
        
        return V

    def _update_value(self, V):
        while True:
            max_delta = 0
            sum_delta = 0

            V_new = np.zeros(self.env.nS) # only used in synchronous case

            for s in range(self.env.nS):
                action_values = self.one_step_lookahead(s, V)
                best_action_value = np.max(action_values)
                
                error = np.abs(best_action_value - V[s])
                max_delta = max(max_delta, error)
                sum_delta += error

                if self.is_synchronous:
                    V_new[s] = best_action_value        
                else:
                    V[s] = best_action_value

            if self.is_synchronous:
                V = V_new
            
            self.delta_list.append(sum_delta / self.env.nS)
            if max_delta < self.theta:
                break
        
        return V

    def value_iteration(self):
        V = np.zeros(self.env.nS)

        if self.is_prioritised:
            V = self._update_value_prioritised(V)
        else:
            V = self._update_value(V)
        
        self.policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            action_values = self.one_step_lookahead(s, V)
            best_action = np.argmax(action_values)

            self.policy[s] = np.eye(self.env.nA)[best_action]
        
        return self.policy, V

    def get_action(self, state):
        return np.argmax(self.policy[state])

    def one_step_lookahead(self, state, V):
            A = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[state][a]:
                    A[a] += prob * \
                        (reward + self.discount_factor * V[next_state])
            return A




def evaluate_random_policy():
    env = GridworldEnv()
    dp_solver = DP()
    dp_solver.set_env(env)

    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = dp_solver.policy_evaluation(random_policy)

    print("Value Function:")
    print(v.reshape(env.shape))
    print("")

def compare_value_iterations():
    env = GridworldEnv(shape=[20, 20])

    # synchronous DP vs. async in-place DP vs. async prioritised DP
    dp_async_inplace = DP(env=env)
    dp_async_prioritised = DP(is_prioritised=True, env=env)
    dp_sync = DP(is_synchronous=True, env=env)

    # Resulting Policy & Value would be same, but the iterations for convergence would differ.

    policy, V = dp_sync.value_iteration()

    print("Grid Policy By Synchronous DP (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    policy, V = dp_async_inplace.value_iteration()

    print("Grid Policy By Synchronous DP (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    policy, V = dp_async_prioritised.value_iteration()
    
    print("Grid Policy By Synchronous DP (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    print(dp_sync.delta_list)
    print(dp_async_inplace.delta_list)
    print(dp_async_prioritised.delta_list)


def _move(env, agent):
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    current_state = env.s  # start point

    print('Start Position: ({}, {})'.format(
        current_state // env.shape[1], current_state % env.shape[1]))

    env._render()

    done = False
    while not done:
        action = agent.get_action(current_state)
        observation, reward, done, info = env.step(action)

        current_state = observation
        print('Action: {}'.format(directions[action]))
        print('Reward: {}'.format(reward))
        print('Done: {}'.format(done))
        env._render()

def move_by_value_iteration():
    env = GridworldEnv()
    dp_solver = DP()
    dp_solver.set_env(env)

    policy, V = dp_solver.value_iteration()

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    _move(env, dp_solver)

def move_by_policy_iteration():
    env = GridworldEnv()
    dp_solver = DP()
    dp_solver.set_env(env)

    policy, V = dp_solver.policy_iteration()

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Grid Value Function:")
    print(V.reshape(env.shape))
    print("")

    print("Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    _move(env, dp_solver)



if __name__ == "__main__":
    # move_by_policy_iteration()
    # move_by_value_iteration()
    compare_value_iterations()
