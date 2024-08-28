import sys

import numpy as np

import madupite as md

numStates = 50
numActions = 4  # Four possible movements: up, right, down, left
maze_width = 10
maze_height = 5

# Define obstacles in the maze
obstacles = {12, 13, 14, 22, 27, 32}  # These states will be obstacles


def rewardfunc(state, action):
    # Simple reward function: -1 for each step unless you reach the goal (state 49)
    if state == numStates // 2:
        return 100  # Goal state reward
    if state in obstacles:
        return -10  # Penalty for hitting an obstacle
    return -1  # Penalty for each move


def probfunc(state, action):
    # Action: 0=Up, 1=Right, 2=Down, 3=Left
    row, col = divmod(state, maze_width)
    next_state = state

    if action == 0 and row > 0:  # Move Up
        next_state -= maze_width
    elif action == 1 and col < maze_width - 1:  # Move Right
        next_state += 1
    elif action == 2 and row < maze_height - 1:  # Move Down
        next_state += maze_width
    elif action == 3 and col > 0:  # Move Left
        next_state -= 1

    if next_state in obstacles:
        next_state = state  # Stay in the same state if moving into an obstacle

    return [1.0], [next_state]  # Deterministic transitions for simplicity


def print_maze(policy):
    symbols = ["↑", "→", "↓", "←"]  # Up, Right, Down, Left
    maze = [[" " for _ in range(maze_width)] for _ in range(maze_height)]

    for state in range(numStates):
        row, col = divmod(state, maze_width)
        if state == numStates // 2:
            maze[row][col] = "G"  # Goal
        elif state in obstacles:
            maze[row][col] = "█"  # Obstacle
        else:
            maze[row][col] = symbols[policy[state]]

    # Create a border around the maze
    top_border = "┌" + "─" * (maze_width * 2 - 1) + "┐"
    bottom_border = "└" + "─" * (maze_width * 2 - 1) + "┘"
    print(top_border)
    for row in maze:
        print("│" + " ".join(row) + "│")
    print(bottom_border)


def main():
    md.initialize_madupite(sys.argv)
    mdp = md.MDP()
    mdp.setOption("-mode", "MAXREWARD")
    mdp.setOption("-discount_factor", "0.99")
    mdp.setOption("-file_policy", "maze_policy.out")
    mdp.setOption("-max_iter_pi", "1000")
    mdp.setOption("-max_iter_ksp", "1000")
    mdp.setOption("-alpha", "1e-4")
    mdp.setOption("-atol_pi", "1e-8")
    mdp.setOption("-default_filenames", "false")

    g = md.createStageCostMatrix(
        name="g", numStates=numStates, numActions=numActions, func=rewardfunc
    )
    P = md.createTransitionProbabilityTensor(
        name="P",
        numStates=numStates,
        numActions=numActions,
        func=probfunc,
    )
    mdp.setStageCostMatrix(g)
    mdp.setTransitionProbabilityTensor(P)
    mdp.solve()

    policy = np.loadtxt("maze_policy.out", dtype=int)

    rank, size = md.mpi_rank_size()
    if rank == 0:
        print(
            f"Successful installation! Solved a MDP with {numStates} states and {numActions} actions in parallel with {size} ranks.",
            flush=True,
        )
        print("\nOptimal Policy (Maze):")
        print_maze(policy)


if __name__ == "__main__":
    main()
