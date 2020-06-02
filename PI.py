from dynamic_programming import DP
from envs.gridworld import GridworldEnv

if __name__ == '__main__':
    env = GridworldEnv([4, 4])

    dp_agent = DP(env=env)
    p, v = dp_agent.policy_iteration()
    print(p)
    print(v)

    dp_agent2 = DP(env=env)
    p, v = dp_agent2.value_iteration()
    print(p)
    print(v)

    dp_agent3 = DP(is_prioritised=True, env=env)
    p, v = dp_agent3.value_iteration()
    print(p)
    print(v)