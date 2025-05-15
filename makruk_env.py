import subprocess
import threading
import queue
import os
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygame
import re


class FairyStockfishMakruk(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    def __init__(self, path="./engine/fairy-stockfish-arm", max_depth=10, render_mode=None, 
                 play_mode="human-vs-ai", human_color='w', engine_timeout=5.0):
        super().__init__()
        self.pygame = pygame
        self.render_mode = render_mode
        self.play_mode = play_mode
        self.human_color = human_color
        self.tile_size = 80
        self.window_size = self.tile_size * 8
        self.engine_timeout = engine_timeout
        self.engine_running = False
        self.piece_count_history = {}  # Track piece counts for Makruk counting rules
        
        if render_mode == "human":
            self.pygame.init()
            self.screen = self.pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = self.pygame.time.Clock()
        self.images = self._load_piece_images()
        
        # Start engine with error handling
        try:
            self.process = subprocess.Popen(
                path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.engine_running = True
        except (FileNotFoundError, PermissionError) as e:
            raise RuntimeError(f"Failed to start Fairy Stockfish engine: {e}")
        
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._enqueue_output)
        self.thread.daemon = True
        self.thread.start()
        
        # Initialize engine with error handling
        try:
            self._send_with_timeout("uci", "uciok")
            self._send("setoption name UCI_Variant value makruk")
            self._send(f"setoption name Threads value {os.cpu_count()}")
            self._send("setoption name Hash value 256")
            self._send_with_timeout("isready", "readyok")
        except TimeoutError as e:
            self.close()
            raise RuntimeError(f"Engine initialization failed: {e}")
        
        self.max_depth = max_depth
        self.done = False
        self.info = {}
        self.uci_moves = self._generate_all_uci_moves()
        # print(len(self.uci_moves))
        self.action_space = spaces.Discrete(len(self.uci_moves))
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 13), dtype=np.int8)
        
        # Makruk-specific variables
        self.move_count = 0
        self.counting_active = False
        self.moves_without_capture = 0
        self.board_history = []  # For threefold repetition detection
        self.consecutive_reversible_moves = 0  # For counting rule
        self.bare_king_countdown = {  # For bare king counting rules
            'w': None,  # Will store move number when white is reduced to bare king
            'b': None   # Will store move number when black is reduced to bare king
        }
    
    def _send_with_timeout(self, command, expected_response=None, timeout=None):
        """Send a command and wait for an expected response with timeout."""
        if timeout is None:
            timeout = self.engine_timeout
            
        if not self.engine_running:
            raise RuntimeError("Engine is not running")
            
        self._send(command)
        
        if expected_response:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    line = self.queue.get(timeout=0.1)
                    if expected_response in line:
                        return line
                except queue.Empty:
                    continue
                    
            raise TimeoutError(f"Timeout waiting for '{expected_response}' after command '{command}'")
    
    def _enqueue_output(self):
        """Thread function to read engine output."""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.queue.put(line.strip())
        except (ValueError, IOError) as e:
            # This happens when the process is terminated
            self.engine_running = False
    
    def _send(self, command: str):
        """Send a command to the engine."""
        if not self.engine_running:
            raise RuntimeError("Engine is not running")
            
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, IOError) as e:
            self.engine_running = False
            raise RuntimeError(f"Failed to send command to engine: {e}")
    
    def _read_line(self, timeout=1.0):
        """Read a line from the engine with timeout."""
        if not self.engine_running:
            return None
            
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _wait_for(self, keyword: str, timeout=None):
        """Wait for a specific keyword in engine output with timeout."""
        if timeout is None:
            timeout = self.engine_timeout
            
        start_time = time.time()
        while self.engine_running and time.time() - start_time < timeout:
            line = self._read_line(timeout=0.1)
            if line and keyword in line:
                return line
        
        if not self.engine_running:
            raise RuntimeError("Engine stopped running while waiting for response")
        else:
            raise TimeoutError(f"Timeout waiting for '{keyword}'")
    
    def _load_piece_images(self):
        piece_symbols = {
            'p': 'black_p.png', 'r': 'black_r.png', 'n': 'black_n.png', 's': 'black_s.png', 'm': 'black_m.png', 'k': 'black_k.png',
            'P': 'white_p.png', 'R': 'white_r.png', 'N': 'white_n.png', 'S': 'white_s.png', 'M': 'white_m.png', 'K': 'white_k.png'
        }
        images = {}
        for symbol, filename in piece_symbols.items():
            path = os.path.join("piece_images", filename)
            images[symbol] = self.pygame.transform.scale(self.pygame.image.load(path), (self.tile_size, self.tile_size))
        return images


    def _generate_all_uci_moves(self):
        files = 'abcdefgh'
        ranks = '12345678'
        base_moves = []
        # 1. All normal 4-character UCI moves
        for f1 in files:
            for r1 in ranks:
                for f2 in files:
                    for r2 in ranks:
                        if f1 + r1 != f2 + r2:
                            base_moves.append(f1 + r1 + f2 + r2)
        # 2. Add promotion moves (Seed to Met = 'm') for pawn promotion squares
        promo_moves = []
        for i, f1 in enumerate(files):
            # White pawn promotes to 6th rank
            promo_moves.append(f1 + '5' + f1 + '6' + 'm')
            promo_moves.append(files[np.clip(i-1,0,7)] + '5' + f1 + '6' + 'm')
            promo_moves.append(files[np.clip(i+1,0,7)] + '5' + f1 + '6' + 'm')

            # Black pawn promotes to 3rd rank
            promo_moves.append(f1 + '4' + f1 + '3' + 'm')
            promo_moves.append(files[np.clip(i-1,0,7)] + '4' + f1 + '3' + 'm')
            promo_moves.append(files[np.clip(i+1,0,7)] + '4' + f1 + '3' + 'm')

        return sorted(list(set(base_moves + promo_moves)))

    def set_position(self, last_move=None):
        if last_move:
            current_fen = self.get_fen()
            self._send(f"position fen {current_fen} moves {last_move}")
        else:
            self._send("position startpos")

    def get_best_move(self, depth=None):
        self._send(f"go depth {depth or self.max_depth}")
        while True:
            line = self._read_line()
            if line and line.startswith("bestmove"):
                return line.split()[1]

    def get_fen(self):
        """Get the current FEN with error handling."""
        try:
            self._send_with_timeout("isready", "readyok")
            self._send("d")
            
            start_time = time.time()
            while time.time() - start_time < self.engine_timeout:
                line = self._read_line(timeout=0.1)
                if line and line.startswith("Fen:"):
                    return line.split("Fen:")[-1].strip()
            
            raise TimeoutError("Timeout getting FEN position")
        except Exception as e:
            # If we can't get the FEN, try to recover
            self._try_engine_recovery()
            raise RuntimeError(f"Failed to get FEN: {e}")
    
    def get_turn(self):
        fen = self.get_fen()
        parts = fen.split()
        return parts[1] if len(parts) > 1 else 'w'

    def _is_capture(self, uci_move):
        """Check if a move is a capture by examining the board before and after."""
        # Get the current FEN before the move
        current_fen = self.get_fen()
        
        # Save the current piece count
        current_pieces = current_fen.count('p') + current_fen.count('P') + \
                        current_fen.count('r') + current_fen.count('R') + \
                        current_fen.count('n') + current_fen.count('N') + \
                        current_fen.count('s') + current_fen.count('S') + \
                        current_fen.count('m') + current_fen.count('M') + \
                        current_fen.count('k') + current_fen.count('K')
        
        # Make the move temporarily
        self._send(f"position fen {current_fen} moves {uci_move}")
        
        # Get the new FEN after the move
        self._send("d")
        new_fen = None
        while True:
            line = self._read_line()
            if line is None:
                break
            if line.startswith("Fen:"):
                new_fen = line.split("Fen:")[-1].strip()
                break
        
        # Count pieces in the new position
        if new_fen:
            new_pieces = new_fen.count('p') + new_fen.count('P') + \
                        new_fen.count('r') + new_fen.count('R') + \
                        new_fen.count('n') + new_fen.count('N') + \
                        new_fen.count('s') + new_fen.count('S') + \
                        new_fen.count('m') + new_fen.count('M') + \
                        new_fen.count('k') + new_fen.count('K')
        else:
            # If we couldn't get the new FEN, assume no capture
            new_pieces = current_pieces
        
        # Restore the original position
        self._send(f"position fen {current_fen}")
        
        # If there are fewer pieces after the move, it was a capture
        return new_pieces < current_pieces

    def _try_engine_recovery(self):
        """Attempt to recover the engine if it's unresponsive."""
        if self.engine_running:
            try:
                # Try a simple command to see if engine responds
                self._send_with_timeout("isready", "readyok", timeout=2.0)
                return True  # Engine responded
            except:
                # Engine is running but not responding, restart it
                self.close()
                time.sleep(1)
                self.__init__(
                    path=self.process.args[0],
                    max_depth=self.max_depth,
                    render_mode=self.render_mode,
                    play_mode=self.play_mode,
                    human_color=self.human_color
                )
                return True
        return False
    
    def get_fen_tensor(self):
        fen_parts = self.get_fen().split()
        board_fen = fen_parts[0]
        turn = fen_parts[1] if len(fen_parts) > 1 else 'w'

        piece_map = {'p': 0, 'r': 1, 'n': 2, 's': 3, 'm': 4, 'k': 5,
                    'P': 6, 'R': 7, 'N': 8, 'S': 9, 'M': 10, 'K': 11}

        tensor = np.zeros((8, 8, 13), dtype=np.int8)
        for i, row in enumerate(board_fen.split('/')):
            col = 0
            for char in row:
                if char.isdigit():
                    col += int(char)
                else:
                    if char in piece_map:
                        tensor[i, col, piece_map[char]] = 1
                    col += 1
        # Set entire 13th channel (index 12) to 1 if it's white's turn
        if turn == 'w':
            tensor[:, :, 12] = 1    # full plane of 1s for white
        return tensor
            
    def update_piece_counts(self):
        """Update piece counts for Makruk counting rules."""
        fen = self.get_fen().split()[0]
        piece_count = {'w': 0, 'b': 0}
        non_king_pieces = {'w': 0, 'b': 0}
        
        # Count pieces
        for char in fen:
            if char in 'RNSMP':  # White pieces excluding king
                piece_count['w'] += 1
                non_king_pieces['w'] += 1
            elif char in 'rnsmp':  # Black pieces excluding king
                piece_count['b'] += 1
                non_king_pieces['b'] += 1
            elif char == 'K':  # White king
                piece_count['w'] += 1
            elif char == 'k':  # Black king
                piece_count['b'] += 1
        
        # Update bare king countdown
        for color in ['w', 'b']:
            if non_king_pieces[color] == 0 and self.bare_king_countdown[color] is None:
                # This side has just been reduced to a bare king
                self.bare_king_countdown[color] = self.move_count
        
        # Store the piece count for this position
        position_key = self.get_fen().split(' ')[0]  # Board position part of FEN
        if position_key in self.piece_count_history:
            self.piece_count_history[position_key] += 1
        else:
            self.piece_count_history[position_key] = 1
            
        return piece_count

    def apply_human_move(self, uci_move):
        if uci_move in self.get_legal_moves():
            self.set_position(uci_move)
            return True
        return False

    def apply_engine_move(self):
        move = self.get_best_move()
        if move:
            self.set_position(move)
        return move

    def render(self):
        if self.render_mode == "human":
            self._render_pygame_board()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array_silent()
        else:
            raise ValueError("Render mode has to be either human or rgb_array")

    def get_legal_moves(self):
        """Get legal moves with better regex pattern for promotion moves."""
        self._send("go perft 1")
        legal_moves = []
        # Updated regex to properly match promotion moves
        move_pattern = re.compile(r'^[a-h][1-8][a-h][1-8][a-z]?$')
        while True:
            line = self._read_line()
            if not line:
                continue
            elif line.startswith("Nodes"):
                break
            else:
                # Extract just the move part before any colon
                if ":" in line:
                    move = line.split(":")[0].strip()
                else:
                    move = line.strip()
                    
                # Check if it matches our pattern
                if move_pattern.match(move):
                    legal_moves.append(move)
        
        return list(set(legal_moves))

    def get_legal_moves_mask(self):
        """Return a binary mask (length = len(self.uci_moves)) indicating legal moves."""
        legal = set(self.get_legal_moves())
        return np.array([1 if move in legal else 0 for move in self.uci_moves], dtype=np.int8)

    def _render_pygame_board(self):
        colors = [self.pygame.Color(240, 217, 181), self.pygame.Color(181, 136, 99)]
        fen = self.get_fen().split()[0]
        board = fen.split('/')
        for row in range(8):
            col_index = 0
            for char in board[row]:
                if char.isdigit():
                    for _ in range(int(char)):
                        rect = self.pygame.Rect(col_index * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size)
                        self.pygame.draw.rect(self.screen, colors[(row + col_index) % 2], rect)
                        col_index += 1
                else:
                    rect = self.pygame.Rect(col_index * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size)
                    self.pygame.draw.rect(self.screen, colors[(row + col_index) % 2], rect)
                    if char in self.images:
                        self.screen.blit(self.images[char], rect)
                    col_index += 1
        self.pygame.display.flip()
        self.clock.tick(10)

    def _render_rgb_array_silent(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_axis_off()
        board = self.get_fen().split()[0].split('/')
        for y, row in enumerate(board):
            x = 0
            for c in row:
                if c.isdigit():
                    x += int(c)
                else:
                    ax.text(x + 0.5, 7.5 - y + 0.5, c, fontsize=16, ha='center', va='center')
                    x += 1
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image
    
    def is_move_irreversible(self, uci_move):
        """Check if a move is irreversible (pawn move or capture)."""
        # Get the piece at the source square
        fen = self.get_fen().split()[0]
        board = []
        for row in fen.split('/'):
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend([''] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)
        
        # Convert UCI coordinates to board indices
        from_file, from_rank = ord(uci_move[0]) - ord('a'), 8 - int(uci_move[1])
        to_file, to_rank = ord(uci_move[2]) - ord('a'), 8 - int(uci_move[3])
        
        # Check if it's a pawn move
        piece = board[from_rank][from_file] if 0 <= from_rank < 8 and 0 <= from_file < 8 else ''
        if piece.lower() == 'p':
            return True
            
        # Check if it's a capture
        target = board[to_rank][to_file] if 0 <= to_rank < 8 and 0 <= to_file < 8 else ''
        if target != '':
            return True
            
        return False
    
    def check_game_result(self):
        """Makruk: checkmate, stalemate, or draw by counting rules (advanced + cowries)."""
        # 1. Check for checkmate/stalemate
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            self._send("d")
            while True:
                line = self._read_line()
                if line is None:
                    break
                if line.startswith("Checkers:"):
                    in_check = bool(line.split("Checkers:")[-1].strip())
                    if in_check:
                        # Disadvantaged side checkmates during active count → draw
                        if self.counting_active and self.bare_king_countdown[self.get_turn()] is None:
                            return True, "draw", "checkmate_by_disadvantaged_while_counting"
                        else:
                            # Return the winner (opposite of current turn)
                            winner = "black" if self.get_turn() == 'w' else "white"
                            return True, "checkmate", {"winner": winner}
                    else:
                        return True, "stalemate", {"result": "draw"}
        
        # 2. Check for 200 moves without capture
        if self.moves_without_capture >= 200:
            return True, "draw", {"result": "draw", "reason": "200_moves_without_capture"}
        
        # 3. Parse FEN board
        fen = self.get_fen().split()[0]
        board = []
        for row in fen.split('/'):
            expanded = []
            for c in row:
                if c.isdigit():
                    expanded.extend(['.'] * int(c))
                else:
                    expanded.append(c)
            board.append(expanded)
        piece_list = [c for row in board for c in row]
        turn = self.get_turn()
        opponent = 'b' if turn == 'w' else 'w'
        
        # 4. Count pawns (cowries)
        white_pawns = piece_list.count('P')
        black_pawns = piece_list.count('p')
        cowries_gone = white_pawns == 0 and black_pawns == 0  # Fixed comparison
        
        # 5. Count non-king, non-pawn pieces
        def piece_counts(color):
            return {
                'r': piece_list.count('r' if color == 'b' else 'R'),
                's': piece_list.count('s' if color == 'b' else 'S'),
                'n': piece_list.count('n' if color == 'b' else 'N'),
                'p': piece_list.count('p' if color == 'b' else 'P')
            }
        
        # 6. Count remaining non-king pieces
        player_counts = piece_counts(turn)
        opponent_counts = piece_counts(opponent)
        # print("Player_counts: ", player_counts)
        # print("Opponent_counts: ", opponent_counts)
        
        # 7. Calculate non-king pieces for both players
        player_non_kings = sum(v for k, v in player_counts.items() if k != 'p')
        opponent_non_kings = sum(v for k, v in opponent_counts.items() if k != 'p')
        
        # 8. Reset countdown if a player regains pieces
        if player_counts['p'] > 0 or player_non_kings > 0:
            if self.bare_king_countdown[turn] is not None:
                # print(f"Player {turn} regained pieces, resetting bare king countdown")
                self.bare_king_countdown[turn] = None
        
        if opponent_counts['p'] > 0 or opponent_non_kings > 0:
            if self.bare_king_countdown[opponent] is not None:
                # print(f"Opponent {opponent} regained pieces, resetting bare king countdown")
                self.bare_king_countdown[opponent] = None
        
        # 9. Detect bare king for opponent
        if opponent_counts['p'] == 0 and opponent_non_kings == 0:  # Fixed comparison
            if self.bare_king_countdown[opponent] is None:
                # print(f"Detected bare king for opponent {opponent} at move {self.move_count}")
                self.bare_king_countdown[opponent] = self.move_count
                # Activate counting when a king becomes bare
                self.counting_active = True
        
        # 10. Debug output
        # print(f"Bare king countdown: {self.bare_king_countdown}")
        # print(f"Counting active: {self.counting_active}")
        # print(f"Current move count: {self.move_count}")
        # print(f"Cowries gone: {cowries_gone}")
        
        # 11. Case 1: pawnless endgame — both sides have no pawns
        if cowries_gone and self.bare_king_countdown[opponent] is not None:
            # Always activate counting for pawnless endgames
            if not self.counting_active:
                self.counting_active = True
                # print("Activating counting for pawnless endgame")
            
            moves_since = self.move_count - self.bare_king_countdown[opponent]
            # print(f"Pawnless endgame: moves since bare king: {moves_since}/64")
            
            if moves_since >= 64:
                print("DRAW TRIGGERED: 64-move cowryless draw")
                return True, "draw", {"result": "draw", "reason": "64_move_cowryless_draw"}
        
        # 12. Case 2: advanced countdown rule based on stronger player's pieces
        if self.bare_king_countdown[opponent] is not None and not cowries_gone:
            moves_since = self.move_count - self.bare_king_countdown[opponent]
            stronger_counts = player_counts
            
            # Fixed all comparisons
            if stronger_counts['r'] >= 2:
                limit = 8
            elif stronger_counts['r'] == 1:
                limit = 16
            elif stronger_counts['r'] == 0 and stronger_counts['s'] >= 2:
                limit = 22
            elif stronger_counts['r'] == 0 and stronger_counts['s'] == 1:  # Fixed comparison
                limit = 44
            elif stronger_counts['r'] == 0 and stronger_counts['s'] == 0 and stronger_counts['n'] >= 2:  # Fixed comparison
                limit = 32
            elif stronger_counts['r'] == 0 and stronger_counts['s'] == 0 and stronger_counts['n'] == 1:  # Fixed comparison
                limit = 64
            elif stronger_counts['r'] == 0 and stronger_counts['s'] == 0 and stronger_counts['n'] == 0 and stronger_counts['p'] > 0:  # Fixed comparison
                limit = 64
            else:
                limit = 64  # fallback default
            
            # print(f"Advanced counting: moves since bare king: {moves_since}/{limit}")
            
            if self.counting_active and moves_since >= limit:
                print(f"DRAW TRIGGERED: counting rule draw after {limit} moves")
                return True, "draw", {"result": "draw", "reason": f"counting_rule_draw_{limit}_moves"}
        
        return False, None, None
        
    def _material_balance(self):
        """
        Simple material count: boat=5, nobleman=3, horse=3, seed=1.
        White positive, Black negative.
        """
        values = {'r':5,'s':3,'n':3,'p':1,'R':5,'S':3,'N':3,'P':1}
        fen = self.get_fen().split()[0]
        balance = 0
        for c in fen:
            if c in values:
                balance += values[c] if c.isupper() else -values[c]
        return balance

    def _in_check(self, color: str) -> bool:
        """
        Return True if the given color ('w' or 'b') is currently in check.
        """
        self._send("d")
        deadline = time.time() + self.engine_timeout
        while time.time() < deadline:
            line = self._read_line(timeout=0.1)
            if line and line.startswith("Checkers:"):
                return line.split("Checkers:")[-1].strip() == "1"
        return False
    
    def _predict_shallow_value(self, uci_move):
        """
        Simulate playing `uci_move`, let the engine reply one ply,
        measure the net material swing, then restore the original position.
        Returns: material_after – material_before.
        """
        # 1) snapshot current position and material
        fen0 = self.get_fen()
        mat0 = self._material_balance()

        # 2) play our move
        self.set_position(last_move=uci_move)

        # 3) engine replies one ply
        opp_move = self.get_best_move()
        self.set_position(last_move=opp_move)

        # 4) measure material again
        mat1 = self._material_balance()

        # 5) restore original
        self._send(f"position fen {fen0}")

        return mat1 - mat0


    def step(self, action):
        # 0) If already done, just return
        if self.done:
            return self.get_fen_tensor(), 0.0, True, False, self.info

        # 1) Decode action & check legality
        uci_move = self.uci_moves[action]
        legal_moves = self.get_legal_moves()
        if uci_move not in legal_moves:
            return self.get_fen_tensor(), 0.0, False, False, {"invalid": True}

        # 2) Snapshot material before agent move
        mat_before = self._material_balance()

        # 3) Apply agent move
        self.set_position(last_move=uci_move)
        self.move_count += 1

        # 4) Capture tracking after agent move
        mat_after_agent = self._material_balance()
        if mat_after_agent != mat_before:
            self.moves_without_capture = 0
        else:
            self.moves_without_capture += 1

        # 5) Check terminal after agent move
        done, result_type, reason = self.check_game_result()
        if done:
            self.done = True
            reward = 1.0 if result_type == "checkmate" else 0.0
            return self.get_fen_tensor(), reward, True, False, {"end_reason": result_type}

        # 6) If self-play, engine replies
        if self.play_mode == "selfplay":
            opp_move = self.get_best_move()
            self.set_position(last_move=opp_move)
            self.move_count += 1

            # Capture tracking after opponent move
            mat_after_opp = self._material_balance()
            if mat_after_opp != mat_after_agent:
                self.moves_without_capture = 0
            else:
                self.moves_without_capture += 1

            done, result_type, reason = self.check_game_result()
            if done:
                self.done = True
                reward = -1.0 if result_type == "checkmate" else 0.0
                return self.get_fen_tensor(), reward, True, False, {"end_reason": result_type}

        # 7) Snapshot material after both moves
        mat_after = self._material_balance()

        # 8) Base shaping: material swing + small living bonus
        reward = 0.1 * (mat_before - mat_after)
        if self.move_count > 10:
            reward -= 0.001

        # 9) Check bonuses for checks
        me   = self.get_turn()
        oppc = 'b' if me == 'w' else 'w'
        if self._in_check(oppc):
            reward += 0.2
        if self._in_check(me):
            reward -= 0.1

        # 10) Mobility shaping
        your_moves = len(self.get_legal_moves())
        orig_fen   = self.get_fen()
        parts      = orig_fen.split()
        parts[1]   = 'b' if parts[1] == 'w' else 'w'        # flip side-to-move
        self._send(f"position fen {' '.join(parts)}")
        opp_moves  = len(self.get_legal_moves())
        self._send(f"position fen {orig_fen}")               # restore
        reward    += 0.01 * (your_moves - opp_moves)

        # build 8×8 board from original FEN (for threat & connectivity)
        board = []
        for row in orig_fen.split()[0].split('/'):
            expanded = []
            for c in row:
                if c.isdigit():
                    expanded.extend(['.'] * int(c))
                else:
                    expanded.append(c)
            board.append(expanded)

        # define neighbor-offsets
        offsets = {
            'P': [(-1, 1), (1, 1)],    'p': [(-1, -1), (1, -1)],
            'N': [(-2, -1),(-1, -2),(1, -2),(2, -1),(2, 1),(1, 2),(-1, 2),(-2, 1)],
            'n': [(-2, -1),(-1, -2),(1, -2),(2, -1),(2, 1),(1, 2),(-1, 2),(-2, 1)],
            'S': [(-1, -1),(1, -1),(-1, 1),(1, 1)], 's': [(-1, -1),(1, -1),(-1, 1),(1, 1)],
            'R': [(-1, 0),(1, 0),(0, -1),(0, 1)],   'r': [(-1, 0),(1, 0),(0, -1),(0, 1)],
            'M': [(-1, 0),(1, 0),(0, -1),(0, 1)],   'm': [(-1, 0),(1, 0),(0, -1),(0, 1)],
            'K': [(-1, -1),(0, -1),(1, -1),(1, 0),(1, 1),(0, 1),(-1, 1),(-1, 0)],
            'k': [(-1, -1),(0, -1),(1, -1),(1, 0),(1, 1),(0, 1),(-1, 1),(-1, 0)],
        }
        my_pieces  = {'P','N','S','R','M','K'} if me == 'w' else {'p','n','s','r','m','k'}
        opp_pieces = {'r','n','s','m','k'}  if me == 'w' else {'R','N','S','M','K'}
        piece_values = {'r':5,'n':3,'s':3,'m':5,'k':0, 'R':5,'N':3,'S':3,'M':5,'K':0}

        # 11) Threat reward: encourage attacking higher-value enemy pieces
        attacked_value = 0
        for y in range(8):
            for x in range(8):
                p = board[y][x]
                if p in my_pieces:
                    for dx, dy in offsets[p]:
                        ty, tx = y + dy, x + dx
                        if 0 <= ty < 8 and 0 <= tx < 8 and board[ty][tx] in opp_pieces:
                            attacked_value += piece_values[board[ty][tx]]
                            break
        reward += 0.01 * attacked_value

        # 12) Connectivity bonus: encourage defended clusters
        conn_bonus = 0
        for y in range(8):
            for x in range(8):
                p = board[y][x]
                if p in my_pieces:
                    for dx, dy in offsets[p]:
                        ty, tx = y + dy, x + dx
                        if 0 <= ty < 8 and 0 <= tx < 8 and board[ty][tx] in my_pieces:
                            conn_bonus += 1
                            break
        reward += 0.01 * conn_bonus

        # 13) Return
        return self.get_fen_tensor(), reward, False, False, {}



    def start_counting(self):
        """Disadvantaged player activates 64-move countdown."""
        self.counting_active = True

    def stop_counting(self):
        """Disadvantaged player stops the countdown voluntarily."""
        self.counting_active = False


    def reset(self, seed=None, options=None):
        """Reset the environment with error handling."""
        try:
            self.done = False
            self.info = {}
            self.move_count = 0
            self.board_history = []
            self.consecutive_reversible_moves = 0
            self.bare_king_countdown = {'w': None, 'b': None}
            self.piece_count_history = {}
            self.moves_without_capture = 0  # Reset moves without capture
            self.counting_active = False  # Reset counting
            self.set_position()
            obs = self.get_fen_tensor()
            return obs, {"legal_moves_mask": self.get_legal_moves_mask()}
        except Exception as e:
            self._try_engine_recovery()
            raise RuntimeError(f"Failed to reset environment: {e}")

    def close(self):
        """Clean up resources with proper error handling."""
        if hasattr(self, 'process') and self.process:
            try:
                self._send("quit")
                if hasattr(self, 'thread') and self.thread:
                    self.thread.join(timeout=2)
                
                # Make sure the process terminates
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            except Exception as e:
                print(f"Engine shutdown failed: {e}")
            finally:
                self.engine_running = False
                
        if self.render_mode == "human" and hasattr(self, 'pygame'):
            try:
                self.pygame.quit()
            except Exception as e:
                print(f"Pygame shutdown failed: {e}")
