import numpy as np


class TensorDP:

    def __init__(self,
                 gamma=1.0,
                 error_tol=1e-5):
        self.gamma = gamma
        self.error_tol = error_tol

        # Following attributes will be set after call "set_env()"

        self.env = None  # environment
        self.policy = None  # policy
        self.ns = None  # Num. states
        self.na = None  # Num. actions
        self.P = None  # Transition tensor
        self.R = None  # Reward tensor

    def set_env(self, env, policy=None):
        self.env = env
        if policy is None:
            self.policy = np.ones([env.nS, env.nA]) / env.nA

        self.ns = env.nS
        self.na = env.nA
        self.P = env.P_tensor  # Rank 3 tensor [num. actions x num. states x num. states]
        self.R = env.R_tensor  # Rank 2 tensor [num. actions x num. states]

        print("Tensor DP agent initialized")
        print("Environment spec:  Num. state = {} | Num. actions = {} ".format(env.nS, env.nA))

    def set_policy(self, policy):
        assert self.policy.shape == policy.shape
        self.policy = policy

    def get_r_pi(self, policy):
        r_pi = (policy * self.R).sum(axis=-1)  # [num. states x 1]
        return r_pi

    def get_p_pi(self, policy):
        p_pi = np.einsum("na,anm->nm", policy, self.P)  # [num. states x num. states]
        return p_pi

    def policy_evaluation(self, policy=None, v_init=None):
        # To-Do : check dimension consistency
        """
        :param policy: policy to evaluate (optional)
        :param v_init: initial value 'guesstimation' (optional)
        :return: v_pi: value function of the input policy
        """
        if policy is None:
            policy = self.policy

        r_pi = self.get_r_pi(policy)  # [num. states x 1]
        p_pi = self.get_p_pi(policy)  # [num. states x num. states]

        if v_init is None:
            v_old = np.zeros(self.ns)
        else:
            v_old = v_init

        while True:
            # perform bellman expectation back
            v_new = r_pi + self.gamma * np.matmul(p_pi, v_old)

            # check convergence
            bellman_error = np.linalg.norm(v_new - v_old)
            if bellman_error <= self.error_tol:
                break
            else:
                v_old = v_new
        return v_new

    def policy_improvement(self, policy=None, v_pi=None):
        if policy is None:
            policy = self.policy

        if v_pi is None:
            v_pi = self.policy_evaluation(policy)

        # Compute Q_pi_(s,a) from V_pi(s)
        r_pi = self.get_r_pi(policy)
        q_pi = r_pi + self.P.dot(v_pi)

        # Greedy improvement
        policy_improved = np.zeros_like(policy)
        policy_improved[np.arange(q_pi.shape[1]), q_pi.argmax(axis=0)] = 1
        return policy_improved
