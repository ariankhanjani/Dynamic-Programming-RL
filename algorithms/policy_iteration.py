import time
import numpy as np

def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """
    Policy Iteration Algorithm for solving MDPs (env.P).
    
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

    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states, dtype=float)

    deltas = []
    iterations = 0

    start = time.perf_counter()

    for _ in range(max_iterations):

        
        # POLICY EVALUATION
        
        while True:
            delta = 0.0

            for s in range(n_states):
                v_old = V[s]
                a = policy[s]

                v_new = 0.0
                for prob, next_state, reward, done in env.P[s][a]:
                    v_new += prob * (reward + gamma * V[next_state] * (not done))

                V[s] = v_new
                delta = max(delta, abs(v_new - v_old))

            deltas.append(delta)

            if delta < theta:
                break

       
        # POLICY IMPROVEMENT
        
        policy_stable = True

        for s in range(n_states):
            old_action = policy[s]

            action_values = np.zeros(n_actions, dtype=float)

            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state] * (not done))

            best_action = np.argmax(action_values)
            policy[s] = best_action

            if best_action != old_action:
                policy_stable = False

        iterations += 1

        if policy_stable:
            break

    time_taken = time.perf_counter() - start

    return policy, V, iterations, time_taken, deltas



if __name__ == "__main__":
    import gym
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    policy, V, iterations, time_taken, deltas = policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=100)
    print("Optimal Value Function:")
    print(V)
    print("\nOptimal Policy:")
    print(policy.reshape((4,4)))  # reshape only for FrozenLake
    print(f"\nTime taken: {time_taken:.4f} seconds over {iterations} iterations")