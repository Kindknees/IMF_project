# gym specific, we simply do a copy paste of what we did in the previous cells, wrapping it in the
# MyEnv class, and train a Proximal Policy Optimisation based agent
from gc import callbacks
import os
import ray
import logging
import wandb
import argparse
import yaml
import random
import numpy as np
import torch 
import pickle

import gym
from gym.spaces import Discrete, Tuple, Dict, Box

from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo, sac  # import the type of agents
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

from ray import tune
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from dotenv import load_dotenv # security keys

from models.mlp import SimpleMlp, ChooseSubstationModel, ChooseActionModel
from models.substation_module import RllibSubsationModule
# from models.hierarchical_agent import HierarchicalAgent, GreedySubModelNoWorker
from grid2op_env.grid_to_gym import Grid_Gym, Grid_Gym_Greedy, HierarchicalGridGym
from experiments.preprocess_config import preprocess_config, get_loader
from experiments.stopper import MaxNotImprovedStopper

from experiments.callback import LogDistributionsCallback
# --- 加入這一行 ---
from experiments.custom_policy import InfrequentUpdatePPOTorchPolicy
# --- 加入結束 ---

load_dotenv()
WANDB_API_KEY = os.environ.get("e75f73b497d281def12eb8bd4794c128172e7391")


logging.basicConfig(
    format='[INFO]: %(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d in \
    function %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO
)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("choose_action"):
            return "choose_action_agent"
        else:
            return "choose_substation_agent"
            
LOCAL_DIR = "log_files"

if __name__ == "__main__":
    random.seed(2137)
    np.random.seed(2137)
    torch.manual_seed(2137)
    ModelCatalog.register_custom_model("fcn", SimpleMlp)
    ModelCatalog.register_custom_model("substation_module", RllibSubsationModule)
    # ModelCatalog.register_custom_model("hierarchical_agent", HierarchicalAgent)
    ModelCatalog.register_custom_model("choose_substation_model", ChooseSubstationModel)
    ModelCatalog.register_custom_model("choose_action_model", ChooseActionModel)

    register_env("Grid_Gym", Grid_Gym)
    register_env("Grid_Gym_Greedy", Grid_Gym_Greedy)
    register_env("HierarchicalGridGym", HierarchicalGridGym)
    ray.shutdown()
    ray.init(ignore_reinit_error=False)

    parser = argparse.ArgumentParser(description="Train an agent on the Grid2Op environment")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Algorithm to use", choices=["ppo", "sac"])
    parser.add_argument("--algorithm_config_path", type=str, default="experiments/ppo/ppo_config.yaml", \
                                                         help="Path to config file for the algorithm")
    parser.add_argument("--use_tune", type=bool, default=True, help="Use Tune to train the agent")
    parser.add_argument("--project_name", type=str, default="testing_callback_grid", help="Name of the to be saved in WandB")
    parser.add_argument("--num_iters", type=int, default=1000, help="Number of iterations to train the agent for.")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers to use for training.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to use for training.")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Number of iterations between checkpoints.")
    parser.add_argument("--group" , type=str, default=None, help="Group to use for training.")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from a checkpoint. If yes, group must be specified.")
    parser.add_argument("--grace_period", type = int, default = 400, help = "Minimum number of timesteps before a trial can be early stopped.")
    parser.add_argument("--num_iters_no_improvement", type = int, default = 200, help = "Minimum number of timesteps before a trial can be early stopped.")
    parser.add_argument("--seed", type = int, default = -1, help = "Seed to use for training.")
    parser.add_argument("--with_opponent", type = bool, default= -1, help = "Whether to use an opponent or not.")

    args = parser.parse_args()

    logging.info("Training the agent with the following parameters:")

    for arg in vars(args):
        logging.info(f"{arg.upper()}: {getattr(args, arg)}")

    config = preprocess_config(yaml.load(open(args.algorithm_config_path), Loader=get_loader()))["tune_config"]
    if args.num_workers != -1: # overwrite config if necessary
        config["num_workers"] = args.num_workers
    if args.seed != -1:
        config["seed"] = args.seed
    if args.with_opponent != -1:
        config["env_config"]["with_opponent"] = True
        config["evaluation_config"]["env_config"]["with_opponent"] = True
    
    print("Config is", config)
    for key,val in config.items():
        print(f"Key {key} has value {val}")
        print("-"*20)

    if args.algorithm == "ppo":
        trainer = ppo.PPOTrainer
    elif args.algorithm == "sac":
        trainer = sac.SACTrainer
    else:
        raise ValueError("Unknown algorithm. Choices are: ppo, sac")
 
    ### Hierarchical specific setup: START
    env_config_train = config["env_config"]
    env_config_val = config["evaluation_config"]["env_config"]

    grid_gym = Grid_Gym(env_config_train)
    

    ### Fixed params (修改後)
    # 保留原有的低層級 (Choose Action) Policy 配置
    # 確保從 YAML 載入的 config 字典存在且有內容
    choose_action_orig_config = config["multiagent"]["policies"].get("choose_action_agent", {}).get("config", {})
    if not choose_action_orig_config:
        logging.warning("Could not find original config for choose_action_agent in YAML. Using empty config.")

    choose_action_policy_tuple = (
        PPOTorchPolicy, # <---- 標準 Policy
        gym.spaces.Dict({
            "action_mask": Box(0, 1, shape=(grid_gym.action_space.n,), dtype=np.float32),
            "regular_obs": grid_gym.observation_space,
            "chosen_substation": Discrete(8) # 假設有 8 個變電站可選
        }),
        grid_gym.action_space,
        choose_action_orig_config # 使用從 YAML 載入的配置
    )

    # 修改中層級 (Choose Substation) Policy 配置
    # 確保從 YAML 載入的 config 字典存在且有內容
    choose_substation_orig_config = config["multiagent"]["policies"].get("choose_substation_agent", {}).get("config", {})
    if not choose_substation_orig_config:
        logging.warning("Could not find original config for choose_substation_agent in YAML. Using empty config.")

    choose_substation_policy_tuple = (
        InfrequentUpdatePPOTorchPolicy, # <---- 使用自訂 Policy
        gym.spaces.Dict({
            "regular_obs": grid_gym.observation_space,
            "chosen_action": Discrete(grid_gym.action_space.n), # 前一個動作
        }),
        Discrete(8), # 假設選擇 8 個變電站之一
        { # Policy 的特定配置
            **choose_substation_orig_config, # 合併 YAML 中的配置
            "update_frequency": 2 # <---- 設置更新頻率為 2 (每 2 次 learn_on_batch 更新一次)
        }
    )

    # 更新 config 中的 multiagent policies
    config["multiagent"]["policies"] = {
        "choose_substation_agent": choose_substation_policy_tuple,
        "choose_action_agent": choose_action_policy_tuple
    }

    config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn
    ### Hierarchical specific setup: END

    if args.use_tune:
        # Limit the number of rows.
        reporter = CLIReporter()
        stopper = CombinedStopper(
            MaximumIterationStopper(max_iter = args.num_iters),
            MaxNotImprovedStopper(metric = "episode_reward_mean",
                                    grace_period = args.grace_period, 
                                    num_iters_no_improvement = args.num_iters_no_improvement,
                                         no_stop_if_val = 5500)
                                    )
        
        analysis = ray.tune.run(
                trainer,
                progress_reporter = reporter,
                config = config,
                name = args.group,
                local_dir= LOCAL_DIR,
                checkpoint_freq=args.checkpoint_freq,
                stop = stopper,
                checkpoint_at_end=True,
                num_samples = args.num_samples,
                callbacks=[WandbLoggerCallback(
                            project=args.project_name,
                            group = args.group,
                            api_key =  WANDB_API_KEY,
                            log_config=True)],
                # loggers= [CustomTBXLogger],
                keep_checkpoints_num = 5,
                checkpoint_score_attr="evaluation/episode_reward_mean",
                verbose = 1,
                resume = args.resume
                )
        ray.shutdown()
    else: # use ray trainer directly
        trainer_object = trainer(env=Grid_Gym,
                 config=config)
        
        for step in range(args.num_iters):
            result = trainer_object.train()
            print(result["episode_len_mean"], flush = True)
            if (step+1) % args.checkpoint_freq == 0:
                checkpoint = trainer_object.save()
                print("checkpoint saved at", checkpoint)
            print("-"*40, flush = True)
    

    