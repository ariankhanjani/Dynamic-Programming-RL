import numpy as np
import matplotlib.pyplot as plt


def evaluate_policy(env, policy, num_episodes=500):
    """
    Runs episodes following the given policy.
    Returns success rate (%).
    """
    successes = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = policy[state]
            state, reward, done, info = env.step(action)

        if reward > 0:
            successes += 1

    return 100 * successes / num_episodes


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


def plot_convergence(deltas_pi, deltas_vi):
    plt.figure(figsize=(7, 5))

    # PI curve — solid blue with markers
    plt.plot(
        deltas_pi,
        label='Policy Iteration',
        linestyle='-',
        linewidth=2,
        marker='o',
        markersize=3
    )

    # VI curve — dashed red line
    plt.plot(
        deltas_vi,
        label='Value Iteration',
        linestyle='--',
        linewidth=2,
        color='red'
    )
    
    plt.xlabel('Iterations / Sweeps')
    plt.ylabel('Max Δ')
    plt.title('Convergence Curves')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    