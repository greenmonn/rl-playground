from tensorized_dp import TensorDP
from envs.gridworld import GridworldEnv

if __name__ == '__main__':
    nx = 5
    ny = 5
    env = GridworldEnv([nx, ny])

    dp_agent = TensorDP()
    dp_agent.set_env(env)

    info = dp_agent.policy_iteration()