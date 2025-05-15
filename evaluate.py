import random
import torch
import numpy as np
from makruk_env import FairyStockfishMakruk
from makruk_a2c import A2CConvNet

# from MODEL_CLASS import YOUR_MODEL  # You should save the model class separately
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO



def masked_softmax(logits, mask):
    logits = logits.masked_fill(mask == 0, -1e9)
    return torch.softmax(logits, dim=1)

def evaluate_vs_engine(model, NUM_EVAL_GAMES=10):
    print("=== Evaluation: Agent vs Engine ===")
    wins, losses, draws = 0, 0, 0

    for game in range(NUM_EVAL_GAMES):
        agent_color = 'w' if game < 5 else 'b'
        max_depth = game+1 if game < 5 else game-4
        env = FairyStockfishMakruk(
            render_mode=None, ### "human", "rgb_array", None
            play_mode="human-vs-ai", ### "human-vs-ai", "selfplay", None
            human_color=agent_color,
            max_depth=max_depth
        )
        obs, info = env.reset()
        if env.render_mode == 'human':
            env.render()
        done = False
        legal_mask = env.get_legal_moves_mask()

        while not done:
            if env.get_turn() == agent_color:
                state = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE) ### Change the observation dimension 8x8x13 >> 13x8x8
                mask = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0).to(DEVICE) 

                with torch.no_grad():
                    logits = model(state) ### Adjust the output if you have more than one
                    probs = masked_softmax(logits, mask) ### Mask the illegal moves
                    action = torch.argmax(probs, dim=1).item() ### Selection actions
                    ### REMEMBER: This code will not always work since everyone has his own model and output

                obs, _, done, _, info = env.step(action)
                if env.render_mode == 'human':
                    env.render()
            else:
                move = env.get_best_move(depth=max_depth)
                if move and move in env.uci_moves:
                    idx = env.uci_moves.index(move)
                    obs, _, done, _, info = env.step(idx)
                else:
                    obs, _, done, _, info = env.step(0)
                if env.render_mode == 'human':
                    env.render()
            legal_mask = env.get_legal_moves_mask()

        winner = info.get("winner")
        if winner == ("white" if agent_color == 'w' else "black"):
            wins += 1
        elif winner == ("black" if agent_color == 'w' else "white"):
            losses += 1
        else:
            draws += 1
        env.close()

    print(f"Wins: {wins} | Losses: {losses} | Draws: {draws}\n")

def evaluate_agent_vs_agent(model1, model2, num_games=2, render=False):
    print("=== Evaluation: Agent1 vs Agent2 ===")
    model1_wins, model2_wins, draws = 0, 0, 0
    
    render_mode = "human" if render else None
    
    for game in range(num_games):
        env = FairyStockfishMakruk(render_mode=render_mode, play_mode="selfplay")
        obs, info = env.reset()
        done = False
        legal_mask = env.get_legal_moves_mask()
        model1_color = 'white' if (game%2)==0 else 'black'

        while not done:
            # Determine which agent's turn it is
            current_color = env.get_turn()
            current_model = model1 if current_color == model1_color else model2
            
            # Convert observation to tensor format expected by the model
            state = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mask = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = current_model(state) ### Adjust the output if you have more than one
                probs = masked_softmax(logits, mask) ### Mask the illegal moves
                action = torch.argmax(probs, dim=1).item() ### Selection actions
                ### REMEMBER: This code will not always work since everyone has his own model and output

                obs, _, done, _, info = env.step(action)
                if render:
                    env.render()
            legal_mask = env.get_legal_moves_mask()

        winner = info.get("winner")
        if winner == ("white" if model1_color == 'white' else "black"):
            model1_wins += 1
        elif winner == ("black" if model1_color == 'white' else "white"):
            model2_wins += 1
        else:
            draws += 1
        env.close()
    print(f"Model 1 Wins: {model1_wins} | Model 2 Wins: {model2_wins} | Draws: {draws}\n")


def evaluate_agent_vs_random(model, num_games=6, render=False):
    print("=== Evaluation: Agent vs Random (Fixed Seed) ===")
    agent_wins, random_wins, draws = 0, 0, 0
    render_mode = "human" if render else None
    
    # Fixed seeds for reproducibility
    seeds = [42, 123, 7890, 54321, 999, 12345]
    
    for game in range(num_games):
        # Set up random seed for this game
        np.random.seed(seeds[game])
        random.seed(seeds[game])
        torch.manual_seed(seeds[game])
        
        env = FairyStockfishMakruk(render_mode=render_mode, play_mode="selfplay")
        obs, info = env.reset(seed=seeds[game])
        done = False
        legal_mask = env.get_legal_moves_mask()
        
        # Agent plays white for first 3 games, black for last 3
        agent_color = 'white' if game < 3 else 'black'
        
        while not done:
            # Determine whose turn it is
            current_color = env.get_turn()
            
            if current_color == agent_color:
                # Agent's turn
                state = torch.tensor(obs.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                mask = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits, _ = model(state)
                    probs = masked_softmax(logits, mask)
                    action = torch.argmax(probs, dim=1).item()
            else:
                # Random player's turn
                legal_moves = np.where(legal_mask == 1)[0]
                action = np.random.choice(legal_moves)
            
            # Take the action
            obs, _, done, _, info = env.step(action)
            if render:
                env.render()
            
            # Update legal moves for next step
            legal_mask = env.get_legal_moves_mask()
        
        # Determine the winner
        winner = info.get("winner")
        if winner == agent_color:
            agent_wins += 1
            result = "Win"
        elif winner is not None:  # There is a winner but it's not the agent
            random_wins += 1
            result = "Loss"
        else:
            draws += 1
            result = "Draw"
        
        print(f"Game {game+1}: Agent as {agent_color} - {result}")
        env.close()
    
    print(f"\nSummary:")
    print(f"Agent Wins: {agent_wins} | Random Wins: {random_wins} | Draws: {draws}")
    print(f"Win Rate: {agent_wins/num_games:.2%}")
    return agent_wins, random_wins, draws


if __name__ == "__main__":
    # --- Constants
    NUM_EVAL_GAMES = 100
    ACTION_SPACE = 4076
    INPUT_SHAPE = (13,8,8)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model1 = MaskablePPO.load("ppo_makruk_pvp.zip")  # Your model
    model2 = PPO.load("ppo_imitation_raw_policy.zip")  # Friend's model


    # --- Load Trained Model 1 (Player 1)
    model1 = YOUR_MODEL(INPUT_SHAPE, ACTION_SPACE).to(DEVICE) # Assume that your model outputs the LOGITS of size ACTION_SPACE
    model1.load_state_dict(torch.load(YOUR_WEIGHT_FILE, map_location=DEVICE, weights_only=True))
    model1.eval()

    # --- Load Trained Model 2 (Player 2)
    model2 = YOUR_MODEL(INPUT_SHAPE, ACTION_SPACE).to(DEVICE) # Assume that your model outputs the LOGITS of size ACTION_SPACE
    model2.load_state_dict(torch.load(YOUR_WEIGHT_FILE, map_location=DEVICE, weights_only=True))
    model2.eval()

    evaluate_vs_engine(model1)
    evaluate_agent_vs_agent()
    # evaluate_human_vs_agent()
