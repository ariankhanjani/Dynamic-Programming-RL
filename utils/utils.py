import numpy as np
import matplotlib.pyplot as plt


def evaluate_policy(env, policy, num_episodes=1000, max_steps=100):
    """
    Evaluates a deterministic policy on a Gymnasium environment.
    
    Works for: FrozenLake-v1, CliffWalking-v0, Taxi-v3
    """
    successes = 0
    total_rewards = []
    total_steps = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and steps < max_steps:
            action = policy[state]  # policy: state -> best action
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        # Success criteria:
        # - For Taxi: episode ended normally (done=True) with correct dropoff
        # - For FrozenLake: reached goal (done=True and not fallen in hole)
        # - For CliffWalking: reached goal (done=True)
        if done and not truncated:  # Normal termination (not timeout)
            successes += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    success_rate = 100 * successes / num_episodes
    avg_steps = np.mean(total_steps)
    
    return success_rate, avg_steps


def plot_heatmap(V, title="State Value Heatmap"):
    """
    Plots a heatmap of state values with:
      - RED color spectrum
      - Value labels inside each grid cell
      - Black borders around cells
    """
    
    size = int(np.sqrt(len(V)))
    V_grid = V.reshape(size, size)

    plt.figure(figsize=(6, 5))
    plt.imshow(V_grid, cmap="Reds")

    # Add borders/grid lines
    plt.grid(which='major', color='black', linewidth=1.5)
    plt.xticks(np.arange(-0.5, size, 1), [])
    plt.yticks(np.arange(-0.5, size, 1), [])
    
    # Add text labels (state values)
    for i in range(size):
        for j in range(size):
            plt.text(
                j, i,
                f"{V_grid[i, j]:.2f}",
                ha='center', va='center',
                color="black",
                fontsize=10,
                fontweight='bold'
            )

    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_convergence(deltas_pi, deltas_vi, env_name="Environment"):
    plt.figure(figsize=(10, 6))
    
    # Policy Iteration: plot per policy evaluation cycle
    plt.plot(
        range(1, len(deltas_pi) + 1),
        deltas_pi,
        label='Policy Iteration',
        linestyle='-',
        linewidth=2.5,
        marker='o',
        markersize=6,
        color='blue'
    )
    
    # Value Iteration: plot per sweep
    plt.plot(
        range(1, len(deltas_vi) + 1),
        deltas_vi,
        label='Value Iteration',
        linestyle='--',
        linewidth=2,
        color='red'
    )
    
    plt.xlabel('Iterations (PI: per policy improvement cycle | VI: per sweep)')
    plt.ylabel('Max Δ (Change in Value Function)')
    plt.title(f'Convergence Comparison: {env_name}')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.yscale('log') # Log scale for better visibility
    plt.tight_layout()
    plt.show()
    