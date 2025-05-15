import numpy as np
from makruk_env import FairyStockfishMakruk  # Adjust if needed

#make build -j OS=macos

def main():
    env = FairyStockfishMakruk(render_mode=None)
    obs, info = env.reset()

    # Extract the turn from the 13th channel (index 12)
    turn_channel = obs[:, :, 12]
    turn_value = turn_channel[0, 0]  # Should be 1 for white, 0 for black
    print("Initial turn:", "White" if turn_value == 1 else "Black")

    # Get legal move mask
    legal_mask = info.get("legal_moves_mask", np.ones(env.action_space.n, dtype=np.int8))

    # Print some legal UCI moves
    legal_moves = [env.uci_moves[i] for i in range(len(legal_mask)) if legal_mask[i] == 1]
    print("Legal moves (first 10):", legal_moves[:10])

    # Take the first legal move (or any random one)
    legal_indices = np.where(legal_mask == 1)[0]
    if legal_indices.size > 0:
        action = int(legal_indices[0])
        uci_move = env.uci_moves[action]
        print("Chosen action index:", action)
        print("Corresponding UCI move:", uci_move)

        obs, reward, terminated, truncated, info = env.step(action)

        print("Step result:")
        print("  Reward:", reward)
        print("  Done:", terminated or truncated)
        print("  Info:", info)
    else:
        print("No legal moves available!")

    env.close()

if __name__ == "__main__":
    main()
