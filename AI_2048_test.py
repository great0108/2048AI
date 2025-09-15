from AI_2048_v2 import AI_2048
from env_2048_v2 import Batch2048EnvFast
import numpy as np
from tqdm import tqdm

def generator():
    while True:
        yield

if __name__ == "__main__":
    env = Batch2048EnvFast(num_envs=1)
    obs, info = env.reset()
    print("Initial boards:")
    print(obs)
    print("Info:", info)
    score = np.zeros((env.num_envs,), dtype=np.float32)

    for _ in tqdm(generator()):
        # actions = env.action_space.sample()
        # obs, reward, terminated, truncated, info = env.step(actions)
        actions = AI_2048.find_best(obs, depth=2 if score[0] < 10000 else 3, use_ray=True)
        obs, reward, terminated, truncated, info = env.step(actions)
        score += reward
        if terminated.all():
            break

    print(score)
    print("Final boards:")
    print(obs)
    for row in obs.swapaxes(0, 1):
        for r in row:
            cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
            print(" ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells))
        print()