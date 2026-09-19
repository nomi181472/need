import gymnasium as gym
import numpy as np
import time


class HumanoidWithAuxRewards(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_state = None
        self.is_first_step = True

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.prev_state = self._extract_state(observation)
        self.is_first_step = True
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_state = self._extract_state(observation)

        # Compute auxiliary reward
        aux_reward = self.compute_auxiliary_reward_humanoid(
            self.prev_state,
            current_state,
            reward,
            self.is_first_step
        )

        # Update state tracking
        self.prev_state = current_state
        self.is_first_step = False

        # Store both rewards in info dict
        info['original_reward'] = reward
        info['auxiliary_reward'] = aux_reward

        return observation, aux_reward, terminated, truncated, info

    def _extract_state(self, observation):
        """
        Extract the required state information from the observation.
        For Humanoid-v4, we need to extract:
        - x position (for forward velocity)
        - z height (for upright bonus)
        - pitch and roll angles (for balance bonus)

        The exact indices depend on the observation space structure.
        """
        # Print observation structure on first call to help debug
        if self.prev_state is None:
            print(f"Observation shape: {observation.shape}")
            # Uncomment to see full observation:
            # print(f"First few observation values: {observation[:10]}")

        # For Humanoid-v4:
        # qpos (position) is at the beginning of the observation
        # Index 0: x-position
        # Index 1: z-position (height)
        # Indices for rotation (quaternion) that can be used to derive pitch/roll
        x_pos = observation[0]
        z_pos = observation[1]  # Height

        # For pitch and roll, we need to use rotation data
        # In Humanoid-v4, the rotation quaternion is typically in the first few indices
        # We'll use simple approximations from the quaternion components
        # These might need adjustment based on exact observation structure
        quat = observation[3:7]  # Assuming quaternion is at indices 3-6

        # Simple approximation of pitch and roll from quaternion
        # This is a simplified calculation and might need refinement
        pitch = 2.0 * (quat[0] * quat[2] - quat[3] * quat[1])
        roll = 2.0 * (quat[0] * quat[1] + quat[2] * quat[3])

        return np.array([x_pos, z_pos, pitch, roll])

    def compute_auxiliary_reward_humanoid(self, state, next_state, primary_reward, is_first_step):
        # Estimate forward velocity (x-axis movement)
        forward_reward = 0.0
        if not is_first_step:
            forward_reward = 0.5 * (next_state[0] - state[0])  # [0] = root x-position

        # Get torso height — index 1 in our extracted state
        torso_z = next_state[1]
        upright_bonus = np.clip((torso_z - 0.8) / (1.4 - 0.8), 0.0, 1.0)  # Normalize between [0.8, 1.4] height

        # Pitch/Roll angle
        pitch = next_state[2]
        roll = next_state[3]
        balance_bonus = 1.0 - (abs(pitch) + abs(roll)) / 2.0
        balance_bonus = np.clip(balance_bonus, 0.0, 1.0)

        # Final reward (scaled)
        aux_reward = (
                + 0.5 * forward_reward
                + 0.3 * upright_bonus
                + 0.2 * balance_bonus
        )

        return primary_reward + aux_reward


def test_humanoid_with_aux_rewards():
    # Create base environment
    try:
        base_env = gym.make("Humanoid-v4", render_mode="human")
    except Exception as e:
        print(f"Error with render mode: {e}")
        print("Trying without render...")
        base_env = gym.make("Humanoid-v4")

    # Wrap with auxiliary rewards
    env = HumanoidWithAuxRewards(base_env)

    # Print action space information
    print(f"Action Space: {env.action_space}")
    print(f"Action Space Shape: {env.action_space.shape}")
    print(f"Action Low Bounds: {env.action_space.low}")
    print(f"Action High Bounds: {env.action_space.high}")

    # Run a test episode
    observation, info = env.reset()

    # Print more detailed observation info to help with debugging
    print(f"\nObservation shape: {observation.shape}")
    print(f"First few observation values: {observation[:10]}")

    # For Humanoid, the observation typically includes:
    # - qpos: position/orientation states
    # - qvel: velocity states
    # - cinert: com-based inertia
    # - cvel: com-based velocity
    # - qfrc_actuator: actuator forces
    # - cfrc_ext: external forces
    print("\nExpected structure of Humanoid-v4 observation:")
    print("- qpos (position/orientation): indices 0-9")
    print("- qvel (velocities): indices 10-35")
    print("- cinert (com-based inertia): indices 36-100")
    print("- cvel (com-based velocity): indices 101-134")
    print("- qfrc_actuator (actuator forces): indices 135-161")
    print("- cfrc_ext (external forces): indices 162-337")

    total_reward = 0
    total_aux_reward = 0
    steps = 0

    print("\nRunning test episode with auxiliary rewards...")

    while True:
        # Sample a random action
        action = env.action_space.sample()

        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)

        # Track rewards
        total_reward += info['original_reward']
        total_aux_reward += reward
        steps += 1

        # Print step information
        if steps % 20 == 0:
            print(f"\nStep {steps}:")
            print(f"Original Reward: {info['original_reward']:.4f}")
            print(f"Auxiliary Reward: {reward:.4f}")

            # Decompose auxiliary reward
            if not env.is_first_step:
                forward_component = 0.5 * 0.5 * (
                            env.prev_state[0] - (env.prev_state[0] - (observation[0] - env.prev_state[0])))
                print(f"Forward Component: {forward_component:.4f}")

            torso_z = env.prev_state[1]
            upright_bonus = np.clip((torso_z - 0.8) / (1.4 - 0.8), 0.0, 1.0)
            print(f"Upright Bonus: {0.3 * upright_bonus:.4f}")

            pitch = env.prev_state[2]
            roll = env.prev_state[3]
            balance_bonus = 1.0 - (abs(pitch) + abs(roll)) / 2.0
            balance_bonus = np.clip(balance_bonus, 0.0, 1.0)
            print(f"Balance Bonus: {0.2 * balance_bonus:.4f}")

            print(f"Current position: x={observation[0]:.2f}, z={observation[1]:.2f}")

        # Check if episode is done
        if terminated or truncated or steps >= 1000:
            break

        # Small delay for rendering
        try:
            time.sleep(0.01)
        except:
            pass

    print(f"\nEpisode complete after {steps} steps")
    print(f"Total Original Reward: {total_reward:.4f}")
    print(f"Total Auxiliary Reward: {total_aux_reward:.4f}")

    # Clean up
    env.close()


def inspect_humanoid_observation():
    """Helper function to understand Humanoid observation space in detail"""
    env = gym.make("Humanoid-v4")
    observation, info = env.reset()

    print(f"Observation Type: {type(observation)}")
    print(f"Observation Shape: {observation.shape}")
    print(f"Observation Space: {env.observation_space}")

    # Print the first few values to understand structure
    print("\nFirst 10 values (position data):")
    print(observation[:10])

    # Print different sections
    sections = [
        ("Position (qpos)", slice(0, 10)),
        ("Velocity (qvel)", slice(10, 36)),
        ("COM-based inertia (cinert)", slice(36, 101)),
        ("COM-based velocity (cvel)", slice(101, 135)),
        ("Actuator forces (qfrc_actuator)", slice(135, 162)),
        ("External forces (cfrc_ext)", slice(162, 338))
    ]

    for name, slice_range in sections:
        values = observation[slice_range]
        print(f"\n{name} (indices {slice_range.start}-{slice_range.stop - 1}):")
        print(f"Min: {np.min(values):.4f}, Max: {np.max(values):.4f}, Mean: {np.mean(values):.4f}")
        print(f"First few values: {values[:5]}")

    # Clean up
    env.close()

    return observation


if __name__ == "__main__":
    print("First, let's inspect the Humanoid observation space:")
    obs = inspect_humanoid_observation()
    print("\n" + "=" * 50 + "\n")
    print("Now testing with auxiliary rewards:")
    test_humanoid_with_aux_rewards()