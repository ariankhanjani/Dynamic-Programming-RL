import time
import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):  
    """
    Value Iteration Algorithm for solving MDPs (env.P).
    
    parameters:
        env         : OpenAI Gym environment with discrete state and action spaces  
        gamma       : discount factor
        theta       : small threshold for stopping criterion
        max_iterations : maximum number of iterations to prevent infinite loops 
        
    returns:
        V           : optimal value function
        policy      : optimal deterministic policy
        deltas      : list of Δ values per sweep for convergence plotting
        time_taken  : time taken to converge
        iterations  : number of sweeps
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states, dtype=float)
    deltas = []
    iterations = 0

    start = time.perf_counter()

    for _ in range(max_iterations):
        delta = 0.0
        new_V = np.zeros_like(V)

        # Bellman optimal backups
        for s in range(n_states):
            action_values = np.zeros(n_actions, dtype=float)

            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
            new_V[s] = np.max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        deltas.append(delta)
        iterations += 1

        if delta < theta:
            break

    time_taken = time.perf_counter() - start

    # Extract greedy policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions, dtype=float)

        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state] * (not done))
        policy[s] = int(np.argmax(action_values))

    return policy, V, iterations, time_taken, deltas 


if __name__ == "__main__":
    import gym
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    policy, V, iterations, time_taken, deltas = value_iteration(env, gamma=0.99, theta=1e-8)
    print("Optimal Value Function:")
    print(V)
    print("\nOptimal Policy:")
    print(policy.reshape((4,4)))  # reshape only for FrozenLake
    print(f"\nTime taken: {time_taken:.4f} seconds over {iterations} iterations")