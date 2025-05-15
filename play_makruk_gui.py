import sys
import time
import numpy as np
import pygame
from makruk_env import FairyStockfishMakruk  # Adjust path if needed


def uci_from_coords(start, end):
    files = 'abcdefgh'
    return files[start[0]] + str(8 - start[1]) + files[end[0]] + str(8 - end[1])


def coords_from_click(pos, tile_size):
    return pos[0] // tile_size, pos[1] // tile_size


def main(play_mode="human-vs-ai", human_color='w', delay=1.0):
    env = FairyStockfishMakruk(render_mode="human", play_mode=play_mode, human_color=human_color)
    obs, info = env.reset()
    selected_square = None

    legal_moves_mask = info.get("legal_moves_mask", np.ones(env.action_space.n, dtype=np.int8))

    # Bot plays first if human is black or in selfplay
    if play_mode == "selfplay" or (play_mode == "human-vs-ai" and env.get_turn() != human_color):
        action = np.random.choice(np.where(legal_moves_mask == 1)[0])
        obs, reward, done, _, info = env.step(action)
        print(f"Bot played: {env.uci_moves[action]}")
        time.sleep(delay)

    running = True
    while running and not env.done:
        env.render()

        if play_mode == "selfplay":
            time.sleep(delay)
            legal_moves_mask = env.get_legal_moves_mask()
            action = np.random.choice(np.where(legal_moves_mask == 1)[0])
            obs, reward, done, _, info = env.step(action)
            print(f"Bot played: {env.uci_moves[action]}")
            if done:
                print(f"Game over: {info.get('end_reason')} | {info.get('detail')}")
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN and env.get_turn() == env.human_color:
                mouse_pos = pygame.mouse.get_pos()
                clicked = coords_from_click(mouse_pos, env.tile_size)

                if selected_square is None:
                    selected_square = clicked
                else:
                    uci_move = uci_from_coords(selected_square, clicked)
                    selected_square = None

                    if uci_move not in env.get_legal_moves():
                        print(f"Illegal move: {uci_move}")
                        continue

                    action = env.uci_moves.index(uci_move)
                    obs, reward, done, _, info = env.step(action)
                    env.render()
                    print(f"You played: {uci_move}")

                    if done:
                        print(f"Game over: {info.get('end_reason')} | {info.get('detail')}")
                        running = False
                        break

                    # Bot's turn
                    if env.play_mode == "human-vs-ai" and env.get_turn() != env.human_color:
                        legal_moves_mask = env.get_legal_moves_mask()
                        action = np.random.choice(np.where(legal_moves_mask == 1)[0])
                        obs, reward, done, _, info = env.step(action)
                        bot_move = env.uci_moves[action]
                        print(f"Bot played: {bot_move}")
                        env.render()
                        if done:
                            print(f"Game over: {info.get('end_reason')} | {info.get('detail')}")
                            running = False
                            break

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human-vs-ai", "selfplay"], default="human-vs-ai")
    parser.add_argument("--human_color", choices=["w", "b"], default="w")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between moves in selfplay mode (in seconds)")
    args = parser.parse_args()

    main(play_mode=args.mode, human_color=args.human_color, delay=args.delay)
