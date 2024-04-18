import gymnasium as gym
# import gymnasium_robotics

example_map = [
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
]

env = gym.make('AntMaze_UMaze-v4')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
