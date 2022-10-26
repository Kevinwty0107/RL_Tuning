from audioop import avg
import numpy as np
from pandas import array
import torch
import gym
import argparse
import os
import sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..'))
from utils import utils
from agents import TD3
from envs.env import LinearFitting
from envs.linear_fitting import Linear_model
from agents import DDPG



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, data, eval_episodes=10):
    eval_env = LinearFitting(data)
    eval_env.reset()
    state = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        for _ in range(100):
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward = avg_reward/eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} parameter: {state}")
    print("---------------------------------------")
    return avg_reward



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str, default='linear', help="benchmark to use")
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG, SAC or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=2e2, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--sample_mode",default="random_local")     # random_local: local + global sampling, random: only global sampling

    # Env Related
    parser.add_argument("--use-terminal-action", type=bool, default=True,
                        help="whether to use terminal action wrapper")
    parser.add_argument("--use-reward-difference", type=bool, default=True,
                        help="whether to use difference in reward")
    parser.add_argument("--reward-scaling", type=str, default='linear',
                        help="use cubic/exponential scaling to encourage risky behaviour")
    parser.add_argument("--denoise-threshold", type=int, default=-1,
                        help="denoise small rewards for numerical stability")
    parser.add_argument("--episode-timesteps", type=int, default=200,
                        help="hard limit on timesteps per episode")
    parser.add_argument("--action-history", type=int, default=40,
                        help="number of history actions to record")
    parser.add_argument("--encoding", type=str, default='ir2vec',
                        help="encoding for bitcode programs")
    parser.add_argument("--record-rollouts", type=str, default="ppo_rollouts",
                        help="location to save rollouts")



    args = parser.parse_args()
    env_name = "LinearFitting"
    file_name = f"{args.policy}_{env_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    L = Linear_model()
    x, y = L.data_generate()

    data = [x,y]

    np.save(f"./results/{file_name}_data", data)

    env = LinearFitting(data)
    # env = gym.make("CartPole-v1")
    # env = gym.make("MountainCarContinuous-v0")

    # Set seeds
    env.reset()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    state_dim = env.observation_space.shape[0]  # type: ignore
    action_dim = env.action_space.shape[0]   # type: ignore
    max_action = float(env.action_space.high[0])  # type: ignore

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    elif args.policy == "DDPG":

        policy = DDPG.DDPG(**kwargs)

    else:
        policy = DDPG.DDPG(**kwargs)
        


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, data)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    MAX_EPI_STEPS = 100

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < MAX_EPI_STEPS else 0

        # Store data in replay buffer

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

        if  episode_timesteps  == MAX_EPI_STEPS: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, data))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
            
    if args.bench_name == "linear":

        y_pred,r,coef,intercept = L.model_fitting(x,y)
        print(coef,intercept)