import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium_env.envs.blokus_env import BlokusEnv
from gymnasium_env.envs.single_agent_blokus_env import SingleAgentBlokusEnv
from src.agents.agent import RandomAgent, SimpleFunctionAgent, MiniMaxAgent, QL_Agent
from src.agents.agent_heuristics import greedy
import argparse
import random 

def parse_args():
    parser = argparse.ArgumentParser(description="Blokus environment configuration")
    parser.add_argument('--board_size', type=int, default=7, help='Size of the board')
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--num_eval_episodes', type=int, default=4, help='Number of evaluation episodes')
    parser.add_argument('--models', '-m', type=str, nargs='+', default=['random', 'random'], help='List of models for players')
    parser.add_argument('--player_turn', type=int, default=1, help='Player turn to start')
    args = parser.parse_args()
    if len(args.models) != args.num_players:
        parser.error(f"The number of models ({len(args.models)}) must match the number of players ({args.num_players})")
    if not (1 <= args.player_turn <= args.num_players):
        parser.error(f"player_turn must be between 1 and {args.num_players}")
    return args

def main():
    args = parse_args()
    # print(args.board_size, args.num_players, args.model1, args.model2, args.num_eval_episodes, args.player_turn)
    hidden_agents = [None for _ in range(len(args.models) + 1)]
    
    model_map = {
        "random": RandomAgent,
        "q_agent": QL_Agent,
        "minimax": lambda name: MiniMaxAgent(name=name, depth=1, board_size=args.board_size),
        "greedy": lambda name: SimpleFunctionAgent(name=name, func=greedy)
    }

    for i, model in enumerate(args.models, start=1):
        if model in model_map:
            agent_class = model_map[model]
            hidden_agents[i] = agent_class(name=f"{model}_hidden_agent_{i}") if not callable(agent_class) else agent_class(name=f"{model}_hidden_agent_{i}")
        else:
            raise ValueError(f"Unknown model: {model}")

    env = SingleAgentBlokusEnv(
        board_size=args.board_size, 
        num_players=args.num_players, 
        player_turn=1, 
        mode="good", 
        render_mode="rgb_array", 
        hidden_agents=hidden_agents
    )

    folder_name = f"blokus_games/{args.num_players}_players"
    model_combinations = "-".join(args.models)
    video_folder = f"{folder_name}/{model_combinations}"

    env = RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix="eval",
        episode_trigger=lambda x: True
    )
    env = RecordEpisodeStatistics(env, buffer_length=args.num_eval_episodes)

    for episode_num in range(args.num_eval_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            actions = obs["possible_actions"]
            action = hidden_agents[args.player_turn].get_action(env, obs)  # replace with actual agent logic based on args.model1 and args.model2
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
    env.close()

    with open(f"{video_folder}/episode_stats.txt", "w") as f:
        f.write(f'Episode time taken: {env.time_queue}\n')
        f.write(f'Episode total rewards: {env.return_queue}\n')
        f.write(f'Episode lengths: {env.length_queue}\n')

if __name__ == "__main__":
    main()