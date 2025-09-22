from AI_2048_v2 import AI_2048
from env_2048_v2 import Batch2048EnvFast
import numpy as np
import ray
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler
from ray.experimental import tqdm_ray
from tqdm import tqdm
from ray.util.placement_group import get_current_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.air import session

@ray.remote(num_cpus=1)
def test(params):
    AI_2048.init(params)
    env = Batch2048EnvFast(num_envs=1)
    obs, info = env.reset()
    score = np.zeros((env.num_envs,), dtype=np.float32)
    while True:
        actions = AI_2048.find_best(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        score += reward
        if terminated[0]:
            break

    return score

def mean_score(params=None):
    AI_2048.init(params)
    pg = get_current_placement_group()
    strat = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True
    )

    refs = [test.options(scheduling_strategy=strat).remote(params)
            for _ in range(params["n"])]
    scores = ray.get(refs)
    score = float(np.array(scores).mean())
    tune.report({"score": score})


if __name__ == "__main__":
    ray.init(num_cpus=32)
    short_dir = lambda t: f"t{t.trial_id}"         # 예: t3aa86733
    short_name = lambda t: f"{t.trainable_name}_{t.trial_id}"

    search_space = {
        "n" : 30,
        "sum_weight" : tune.uniform(0, 100),
        "sum_power" : tune.uniform(2, 5),
        "locate_power" : 0,
        "locate_weight" : np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "monotonic_power" : tune.uniform(2, 5),
        "monotonic_weight" : tune.uniform(0, 100),
        "merge_weight" : tune.uniform(0, 100),
        "empty_weight" : tune.uniform(0, 100),
    }

    algo = OptunaSearch(
        sampler=TPESampler(
            n_startup_trials=20,    # 초반 랜덤 탐색
            multivariate=True,      # 파라미터 상관관계 활용
        ),
    )

    N = 30  # 동시에 띄울 test 태스크 수(= 예약할 워커 슬롯 수)
    pg = tune.PlacementGroupFactory(
        [{'CPU': 1.0}] + [{'CPU': 1.0}] * N
    )
    trainable_with_resources = tune.with_resources(mean_score, pg)
    
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            search_alg=algo,
            trial_dirname_creator=short_dir,       # 폴더명 단축
            trial_name_creator=short_name,          # (선택) 화면용 이름 단축
            max_concurrent_trials=1,
            num_samples=100
        ),
        param_space=search_space
    )

    tuner.fit()