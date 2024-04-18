import gymnasium as gym
from gymnasium.utils.save_video import save_video
# import gymnasium_robotics

cross_map = [
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
]

env = gym.make('AntMaze_UMaze-v4', maze_map=cross_map, render_mode="rgb_array")

_ = env.reset()
step_start_i = 0
episode_i = 0
for step_i in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    _, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        save_video(
            env.render(),
            "videos",
            fps=env.metadata["render_fps"],
            step_starting_index=step_starting_index,
            episode_index = episode_index,
        )
        step_start_i = step_i + 1
        episode_i += 1
        observation, info = env.reset()

env.close()
