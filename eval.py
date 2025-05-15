import random
import numpy as np
import torch
from makruk_env import FairyStockfishMakruk
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

# Detect MPS
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def safe_predict(model, obs, mask):
    """
    Try maskable predict; on failure, fall back to plain predict 
    and then force a legal move if needed.
    """
    try:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    except Exception:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        if mask[action] == 0:
            legal = np.where(mask == 1)[0]
            action = int(np.random.choice(legal))
    return int(action)

def evaluate_vs_engine(agent, num_games=10):
    print("=== Evaluation: Agent vs Engine ===")
    wins = losses = draws = 0
    half = num_games // 2

    for game in range(num_games):
        # alternate playing white/black
        human_color = 'w' if game < half else 'b'
        agent_color = "white" if human_color=='w' else "black"
        depth = (game+1) if game<half else (game+1-half)

        env = FairyStockfishMakruk(
            render_mode=None,
            play_mode="human-vs-ai",
            human_color=human_color,
            max_depth=depth
        )
        obs, _ = env.reset()
        done = False

        while not done:
            if env.get_turn() == human_color:
                mask = env.get_legal_moves_mask()
                a = safe_predict(agent, obs, mask)
            else:
                mv = env.get_best_move(depth=depth)
                a = env.uci_moves.index(mv) if mv in env.uci_moves else 0

            obs, _, done, _, info = env.step(a)

        winner = info.get("winner")
        if winner == agent_color:
            wins += 1;  result = "Win"
        elif winner is not None:
            losses += 1; result = "Loss"
        else:
            draws += 1;  result = "Draw"

        print(f"Game {game+1} (agent={agent_color}, depth={depth}): {result}")
        env.close()

    print(f"\nSummary vs Engine → Wins={wins}  Losses={losses}  Draws={draws}\n")

def evaluate_agent_vs_agent(agent1, agent2, num_games=10, render=False):
    print("=== Evaluation: Agent1 vs Agent2 ===")
    a1_wins = a2_wins = draws = 0

    for game in range(num_games):
        m1_color = "white" if (game % 2)==0 else "black"
        env = FairyStockfishMakruk(
            render_mode="human" if render else None,
            play_mode="selfplay"
        )
        obs, _ = env.reset()
        done = False

        while not done:
            turn = env.get_turn()  # 'w' or 'b'
            mask = env.get_legal_moves_mask()
            if (turn=='w' and m1_color=="white") or (turn=='b' and m1_color=="black"):
                a = safe_predict(agent1, obs, mask)
            else:
                a = safe_predict(agent2, obs, mask)

            obs, _, done, _, info = env.step(a)
            if render:
                env.render()

        winner = info.get("winner")
        if winner == m1_color:
            a1_wins += 1;  result = "Agent1 Win"
        elif winner is not None:
            a2_wins += 1;  result = "Agent2 Win"
        else:
            draws += 1;   result = "Draw"

        print(f"Game {game+1} (Agent1 was {m1_color}): {result}")
        env.close()

    print(f"\nSummary Agent1 vs Agent2 → Agent1 Wins={a1_wins}  Agent2 Wins={a2_wins}  Draws={draws}\n")

def evaluate_agent_vs_random(agent, num_games=6, render=False):
    print("=== Evaluation: Agent vs Random ===")
    ag_w, rn_w, dr = 0, 0, 0
    seeds = [42, 123, 7890, 54321, 999, 12345]
    half = num_games // 2

    for game in range(num_games):
        np.random.seed(seeds[game]); random.seed(seeds[game])
        env = FairyStockfishMakruk(
            render_mode="human" if render else None,
            play_mode="selfplay"
        )
        obs, _ = env.reset()
        done = False
        ag_color = "white" if game<half else "black"

        while not done:
            turn = env.get_turn()
            mask = env.get_legal_moves_mask()
            if (turn=='w' and ag_color=="white") or (turn=='b' and ag_color=="black"):
                a = safe_predict(agent, obs, mask)
            else:
                legal = np.where(mask==1)[0]
                a = int(np.random.choice(legal))

            obs, _, done, _, info = env.step(a)
            if render:
                env.render()

        winner = info.get("winner")
        if winner == ag_color:
            ag_w += 1;  res = "Win"
        elif winner is not None:
            rn_w += 1;  res = "Loss"
        else:
            dr += 1;   res = "Draw"

        print(f"Game {game+1} (Agent as {ag_color}): {res}")
        env.close()

    print(f"\nSummary vs Random → Agent Wins={ag_w}  Random Wins={rn_w}  Draws={dr}")
    print(f"Win Rate: {ag_w/num_games:.2%}\n")
    return ag_w, rn_w, dr

if __name__ == "__main__":
    AGENT1_ZIP = "ppo_makruk_pvp.zip"            # your MaskablePPO weights
    AGENT2_ZIP = "ppo_imitation_raw_policy.zip"  # any plain PPO/DQN/etc.

    # load both onto MPS (or CPU)
    agent1 = MaskablePPO.load(AGENT1_ZIP, device=DEVICE)
    agent2 = PPO.load(AGENT2_ZIP, device=DEVICE)

    # evaluate_vs_engine(agent1, num_games=100)
    evaluate_agent_vs_agent(agent1, agent2, num_games=100, render="human")
    # evaluate_agent_vs_random(agent1, num_games=6, render=False)
