from AI_2048_v2 import find_best
from env_2048_v2 import Batch2048EnvFast
from simulate_2048_v2 import Batch2048EnvSimulator
import numpy as np
import ray
from tqdm import tqdm

def generator():
    while True:
        yield

if __name__ == "__main__":
    env = Batch2048EnvFast(num_envs=1, seed=42)
    obs, info = env.reset()
    print("Initial boards:")
    print(obs)
    print("Info:", info)
    score = np.zeros((env.num_envs,), dtype=np.float32)

    for _ in tqdm(generator()):
        # actions = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(actions)
        actions = find_best(obs, depth=2)
        obs, reward, terminated, truncated, info = env.step(actions)
        score += reward
        if terminated.all():
            break

    print(score)
    print("Final boards:")
    print(obs)