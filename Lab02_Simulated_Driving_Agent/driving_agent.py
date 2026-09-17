"""
Lab 02 — Simulated Driving Agent Behavior
==========================================
Build a self-driving agent using Genetic Algorithms in a grid world.

Key Concepts:
- Genetic Algorithms (Selection, Crossover, Mutation)
- Agent-based Simulation
- Fitness Function Design
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ============================================================
# 1. Environment Setup
# ============================================================
print("=" * 60)
print("LAB 02 — Simulated Driving Agent (Genetic Algorithm)")
print("=" * 60)

# Grid world parameters
GRID_ROWS = 10
GRID_COLS = 10
START = (0, 0)
GOAL = (9, 9)
NUM_OBSTACLES = 15
GENOME_LENGTH = 30  # Number of moves per agent
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1

# Possible moves: Up, Down, Left, Right
MOVES = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
}
MOVE_NAMES = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}

# Generate obstacles randomly
np.random.seed(42)
obstacles = set()
while len(obstacles) < NUM_OBSTACLES:
    obs = (random.randint(0, GRID_ROWS - 1), random.randint(0, GRID_COLS - 1))
    if obs != START and obs != GOAL:
        obstacles.add(obs)

print(f"\n[MAP]  Grid Size: {GRID_ROWS} x {GRID_COLS}")
print(f"[CAR] Start: {START}  |  [GOAL] Goal: {GOAL}")
print(f"[WALL] Obstacles: {len(obstacles)}")


# ============================================================
# 2. Helper Functions
# ============================================================
def create_individual():
    """Create a random genome (sequence of moves)."""
    return [random.randint(0, 3) for _ in range(GENOME_LENGTH)]


def simulate_agent(genome):
    """Simulate an agent's path through the grid and return the path."""
    pos = list(START)
    path = [tuple(pos)]

    for move in genome:
        dr, dc = MOVES[move]
        new_r = pos[0] + dr
        new_c = pos[1] + dc

        # Check bounds and obstacles
        if (0 <= new_r < GRID_ROWS and 0 <= new_c < GRID_COLS
                and (new_r, new_c) not in obstacles):
            pos = [new_r, new_c]
            path.append(tuple(pos))

        # Stop if goal reached
        if tuple(pos) == GOAL:
            break

    return path


def fitness(genome):
    """Calculate fitness based on distance to goal and path efficiency."""
    path = simulate_agent(genome)
    final_pos = path[-1]

    # Manhattan distance from final position to goal
    distance = abs(final_pos[0] - GOAL[0]) + abs(final_pos[1] - GOAL[1])

    # Fitness: higher is better
    # Reward for being close to goal, bonus for reaching it
    fit = 1.0 / (1.0 + distance)

    if final_pos == GOAL:
        fit += 10.0  # Big bonus for reaching goal
        fit += 1.0 / len(path)  # Bonus for shorter path

    return fit


def selection(population, fitnesses):
    """Tournament selection."""
    tournament_size = 5
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0][:])  # Copy the genome
    return selected


def crossover(parent1, parent2):
    """Single-point crossover."""
    point = random.randint(1, GENOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(genome):
    """Random mutation of moves."""
    for i in range(len(genome)):
        if random.random() < MUTATION_RATE:
            genome[i] = random.randint(0, 3)
    return genome


# ============================================================
# 3. Genetic Algorithm
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: Running Genetic Algorithm")
print("-" * 40)

# Initialize population
population = [create_individual() for _ in range(POPULATION_SIZE)]

best_fitness_history = []
avg_fitness_history = []

for gen in range(GENERATIONS):
    # Evaluate fitness
    fitnesses = [fitness(ind) for ind in population]

    best_fit = max(fitnesses)
    avg_fit = np.mean(fitnesses)
    best_fitness_history.append(best_fit)
    avg_fitness_history.append(avg_fit)

    if gen % 10 == 0 or gen == GENERATIONS - 1:
        print(f"  Generation {gen:3d} | Best Fitness: {best_fit:.4f} | Avg Fitness: {avg_fit:.4f}")

    # Check if goal reached
    best_idx = np.argmax(fitnesses)
    best_path = simulate_agent(population[best_idx])
    if best_path[-1] == GOAL and gen > 0:
        print(f"  [TARGET] Goal reached at generation {gen}!")

    # Selection
    selected = selection(population, fitnesses)

    # Crossover
    next_gen = []
    for i in range(0, POPULATION_SIZE - 1, 2):
        child1, child2 = crossover(selected[i], selected[i + 1])
        next_gen.extend([child1, child2])

    if len(next_gen) < POPULATION_SIZE:
        next_gen.append(selected[-1][:])

    # Mutation
    population = [mutate(ind) for ind in next_gen]

# Final evaluation
fitnesses = [fitness(ind) for ind in population]
best_idx = np.argmax(fitnesses)
best_genome = population[best_idx]
best_path = simulate_agent(best_genome)

print(f"\n[BEST] Best Agent:")
print(f"   Final Position: {best_path[-1]}")
print(f"   Path Length: {len(best_path)} steps")
print(f"   Reached Goal: {'[OK] Yes' if best_path[-1] == GOAL else '[X] No'}")

# ============================================================
# 4. Visualization — Grid World with Best Path
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Visualization")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot 1: Grid World ---
ax = axes[0]
grid = np.zeros((GRID_ROWS, GRID_COLS))

# Mark obstacles
for obs in obstacles:
    grid[obs[0], obs[1]] = -1

# Mark path
for p in best_path:
    grid[p[0], p[1]] = 0.5

grid[START[0], START[1]] = 0.8
grid[GOAL[0], GOAL[1]] = 1.0

cmap = plt.cm.RdYlGn
ax.imshow(grid, cmap=cmap, origin='upper')

# Draw grid lines
for i in range(GRID_ROWS + 1):
    ax.axhline(i - 0.5, color='gray', linewidth=0.5)
for j in range(GRID_COLS + 1):
    ax.axvline(j - 0.5, color='gray', linewidth=0.5)

# Mark start and goal
ax.text(START[1], START[0], '[CAR]', fontsize=16, ha='center', va='center')
ax.text(GOAL[1], GOAL[0], '[GOAL]', fontsize=16, ha='center', va='center')

# Mark obstacles
for obs in obstacles:
    ax.text(obs[1], obs[0], '[WALL]', fontsize=12, ha='center', va='center')

ax.set_title('Grid World — Best Agent Path', fontsize=14, fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

# --- Plot 2: Fitness Convergence ---
ax2 = axes[1]
ax2.plot(best_fitness_history, label='Best Fitness', color='green', linewidth=2)
ax2.plot(avg_fitness_history, label='Avg Fitness', color='orange', linewidth=2, linestyle='--')
ax2.set_xlabel('Generation')
ax2.set_ylabel('Fitness')
ax2.set_title('Fitness Convergence Over Generations', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('driving_agent_results.png', dpi=150)
plt.show()
print("[OK] Results saved as 'driving_agent_results.png'")

print("\n" + "=" * 60)
print("[OK] Lab 02 Complete!")
print("=" * 60)
