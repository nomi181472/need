import gymnasium as gym
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import base, creator, tools, algorithms


# Step 1: Define Neural Network Policy (for discrete actions)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # Output logits for 4 discrete actions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def sample_action(self, logits: torch.Tensor) -> int:
        """Sample discrete action from logits."""
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()


# Step 2: Define Archive (2D for LunarLander)
class GridArchive:
    def __init__(self, bd_ranges: List[Tuple[float, float]], grid_size: int):
        """
        Initialize a 2D grid archive for LunarLander (x, y)-position.
        bd_ranges: [(min_x, max_x), (min_y, max_y)].
        grid_size: Number of cells per dimension.
        """
        self.min_bd = [r[0] for r in bd_ranges]  # [min_x, min_y]
        self.max_bd = [r[1] for r in bd_ranges]  # [max_x, max_y]
        self.grid_size = grid_size
        self.cell_width = [(max_b - min_b) / grid_size for max_b, min_b in zip(self.max_bd, self.min_bd)]
        self.archive = {}  # Dictionary: (x_idx, y_idx) -> (solution, fitness)

    def get_cell_index(self, bd: Tuple[float, float]) -> Tuple[int, int]:
        """Map 2D behavior descriptor to cell index."""
        x, y = bd
        x_idx = int(np.clip((x - self.min_bd[0]) / self.cell_width[0], 0, self.grid_size - 1))
        y_idx = int(np.clip((y - self.min_bd[1]) / self.cell_width[1], 0, self.grid_size - 1))
        return (x_idx, y_idx)

    def add_solution(self, bd: Tuple[float, float], fitness: float, solution: np.ndarray) -> bool:
        """Add or replace solution in archive if it has higher fitness."""
        cell_index = self.get_cell_index(bd)
        if cell_index not in self.archive or fitness > self.archive[cell_index][1]:
            self.archive[cell_index] = (solution, fitness)
            return True
        return False

    def get_qd_score(self) -> float:
        """Calculate QD score as sum of fitness values in archive."""
        return sum(fitness for _, fitness in self.archive.values())


# Step 3: Evaluate Solution
def evaluate_solution(solution: np.ndarray, env_name: str, policy: PolicyNetwork) -> Tuple[float, Tuple[float, float]]:
    """
    Evaluate a neural network in LunarLander-v3.
    Returns: (fitness, behavior_descriptor).
    """
    env = gym.make(env_name)
    # Set neural network weights
    weights = np.array(solution)
    start_idx = 0
    for param in policy.parameters():
        param_shape = param.data.shape
        param_size = np.prod(param_shape)
        param.data = torch.tensor(weights[start_idx:start_idx + param_size].reshape(param_shape), dtype=torch.float32)
        start_idx += param_size

    # Run episode
    obs, info = env.reset()  # Unpack tuple
    total_reward = 0.0
    final_position = (0.0, 0.0)  # (x, y)-position
    done = False
    while not done:
        # Convert observation to NumPy array and then to tensor
        obs_array = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
        logits = policy(obs_tensor)
        action = policy.sample_action(logits)  # Sample discrete action
        obs, reward, terminated, truncated, info = env.step(action)  # Unpack full return
        total_reward += reward
        final_position = (obs[0], obs[1])  # x, y position
        done = terminated or truncated

    # Fitness: Total reward (higher is better)
    fitness = total_reward
    # Behavior descriptor: Final (x, y)-position
    bd = final_position
    env.close()
    return fitness, bd


# Step 4: Neuroevolution with NSGA-SM
def neuroevolution(env_name: str, pop_size: int, generations: int, archive: GridArchive):
    """Run NSGA-SM neuroevolution and calculate QD score with logging."""
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]  # 8 for LunarLander-v3
    output_dim = env.action_space.n  # 4 discrete actions
    policy = PolicyNetwork(input_dim, output_dim)

    # Calculate total number of weights
    total_weights = sum(np.prod(param.shape) for param in policy.parameters())

    # DEAP setup
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,))  # Single fitness for simplicity
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=total_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda ind: evaluate_solution(ind, env_name, policy))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Evolution loop with logging
    for gen in range(generations):
        # Evaluate population
        fitnesses = []
        for ind in population:
            fitness, bd = toolbox.evaluate(ind)
            ind.fitness.values = (fitness,)
            archive.add_solution(bd, fitness, np.array(ind))
            fitnesses.append(fitness)

        # Compute statistics
        avg_fitness = np.mean(fitnesses)
        max_fitness = np.max(fitnesses)
        archive_size = len(archive.archive)
        qd_score = archive.get_qd_score()

        # Log generation stats
        print(f"Generation {gen + 1}/{generations}:")
        print(f"  Average Fitness: {avg_fitness:.2f}")
        print(f"  Max Fitness: {max_fitness:.2f}")
        print(f"  Archive Size: {archive_size}")
        print(f"  QD Score: {qd_score:.2f}")
        print("-" * 50)

        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation (NSGA-SM: single mutation)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if np.random.random() < 0.1:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring

    # Calculate and log final QD score
    final_qd_score = archive.get_qd_score()
    print(f"Final QD Score: {final_qd_score:.2f}")


# Step 5: Run
if __name__ == "__main__":
    env_name = "LunarLander-v3"
    bd_ranges = [(-1.5, 1.5), (-1.5, 1.5)]  # Range of (x, y)-position
    archive = GridArchive(bd_ranges, grid_size=50)  # 50x50 grid
    neuroevolution(env_name, pop_size=100, generations=100, archive=archive)