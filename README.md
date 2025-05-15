# Makruk RL Agent

This repository contains our Deep Reinforcement Learning project for playing **Makruk** (Thai Chess) using a masked-action PPO agent.

---

## 📦 Project Structure

```
├── engine/                           # Fairy-Stockfish Makruk engine binary
├── makruk_env.py                    # Gymnasium-compatible Makruk environment
├── selfplay_train.py                # Script: self-play training loop
├── train_against_engine.py          # Script: train against engine at varying depths
├── eval.py                          # Evaluation scripts (agent vs engine / agent vs agent / agent vs random)
├── checkpoints/                     # Automatic checkpoint dumps
├── best_model/                      # Saved “best” models
├── ppo_makruk_self_pvp.zip          # Final self-play agent weights
├── ppo_makruk_pvp.zip               # Final agent-vs-engine agent weights
└── README.md                        # This file
```

---

## 🛠️ Dependencies

- Python 3.8+
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (>=1.8.0)
- [sb3_contrib (MaskablePPO)](https://github.com/Stable-Baselines-Team/sb3-contrib)
- Gymnasium
- PyTorch (with CPU or MPS support)
- NumPy, Matplotlib, Pygame
- Fairy-Stockfish Makruk binary (included under `engine/`)

Install via:

```bash
pip install stable-baselines3 sb3-contrib gymnasium torch numpy matplotlib pygame
```

---

## 🚀 Getting Started

### 1. Training

#### a) Against Engine

```bash
python train_against_engine.py   --depth 1   --total-timesteps 300000   --save-path ppo_makruk_pvp.zip
```

This runs PPO with a maskable environment vs. Fairy-Stockfish at `max_depth=1`, saving checkpoints every 50 k steps and the final weights to `ppo_makruk_pvp.zip`.

#### b) Self-Play

```bash
python selfplay_train.py   --current-model ppo_makruk_self_pvp.zip   --best-model ppo_makruk_pvp.zip   --total-timesteps 200000   --save-current ppo_makruk_self_pvp.zip
```

Starts from your engine-trained agent (`ppo_makruk_pvp.zip`), then alternates learning vs. its frozen “best” copy. Checkpoints and best-model snapshots are in `checkpoints/`.

> **Tip:** After finishing engine-vs-agent and self-play at depth 1, you can bump `max_depth=2` in both scripts and repeat.

---

### 2. Inference & Evaluation

Run **eval.py** to evaluate:

- **Agent vs Engine**  
- **Agent vs Agent** (e.g. your PPO vs. a peer’s PPO)  
- **Agent vs Random**  

```bash
python eval.py
```

You can tweak:

- `NUM_EVAL_GAMES`  
- `max_depth` in the engine  
- Which checkpoints to load for `agent1` and `agent2`

---

## 🔍 Reward Function

We use a shaped reward to encourage both defensive and offensive play:

1. **Terminal:**  
   - +1 for a checkmate win  
   - –1 for a checkmate loss  
   - 0 for draws (stalemate or counting-rule draw)

2. **Material swing:**  
   \+0.1 × (material_before – material_after)

3. **Living penalty:**  
   –0.001 per move after move 10

4. **Check bonuses:**  
   +0.2 if opponent is in check after your move  
   –0.1 if you are in check

5. **Mobility:**  
   +0.01 × (your legal moves – opponent’s legal moves)

6. **Connectivity:**  
   +0.01 per “defended” piece (adjacent friendly support)

7. **Capture tracking:**  
   Resets a `moves_without_capture` counter to implement a 200-move no-capture draw rule.

---

## 📊 TensorBoard Logs

- **Engine training** logs in `ppo_makruk_tb/`  
- **Self-play** logs in `ppo_selfplay_tb/`

Key scalars:

- `rollout/ep_rew_mean` – average episode reward  
- `rollout/ep_len_mean` – average episode length  
- `train/learning_rate` – decaying LR  
- `train/entropy_loss` – encourages exploration  
- **Comparison:**  
  - Engine-only shows steady initial rise in reward, slower convergence.  
  - Self-play shows more oscillation but ultimately higher mean return once both sides co-evolve.

---