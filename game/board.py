import copy
from enum import IntEnum
import math
from game.action import Action
from game.direction import Direction
from game.utils import Vec2
from game.tetromino import Rotation, Tetromino, TetrominoType
from game.tetromino_queue import TetrominoQueue

import pygame


class ScoreType(IntEnum):
    Single = 0
    Double = 1
    Triple = 2
    Tetris = 3
    MiniTSpin = 4
    TSpinNoline = 5
    MiniTSpinSingle = 6
    TSpinSingle = 7
    MiniTSpinDouble = 8
    TSpinDouble = 9
    TSpinTriple = 10
    BackToBack = 11
    Combo = 12
    SoftDrop = 13
    HardDrop = 14


class Board:
    TILE_SIZE = 20

    WIDTH = 10
    HEIGHT = 20

    TICK_DT = 1000

    LOCK_DELAY_MS = 500
    MOVE_LOCK_DELAY_LIMIT = 15

    def __init__(self):
        self.matrix = [[None for _ in range(Board.WIDTH)] for _ in range(Board.HEIGHT)]
        self.queue = TetrominoQueue()

        self.hold_piece = None
        self.current = self.queue.next()
        self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

        self.did_swap_current_piece = False
        self.did_turn_last_move = False
        self.did_turn_move_2_side_1_down = False

        self.fall_dt = 0
        self.move_count_fall = 0
        self.fall_move_dt = 0
        self.last_tick_ms = 0
        self.last_tick_ms_score = 0
        self.level = 1
        self.score_level = 0
        self.score = 0
        self.last_score_type = ""
        self.score_alpha = 0
        self.combo = 0
        self.terminated = False

    def try_turn_current(self, direction):
        new_rotation = Rotation.rotate(self.current.rotation, direction)
        kick_tests = self.current.get_wall_kick_tests(new_rotation)

        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.turn(direction)

        new_tetromino = self.test_wall_kicks(kick_tests, tetromino_copy)
        if new_tetromino is not None:
            self.did_turn_last_move = True
            self.current = new_tetromino
            # TODO: Reset counters

    def test_wall_kicks(self, kick_tests, tetromino):
        for i, test in enumerate(kick_tests):
            tetromino_copy = copy.deepcopy(tetromino)
            tetromino_copy.move_relative(test)
            if not self.check_collision(tetromino_copy):
                self.did_turn_move_2_side_1_down = i == 3
                return tetromino_copy

        return None

    def calcular_altura(self):
        matriz = self.current_state()
        altura = 0
        for linha in range(20):
            for coluna in range(10):
                if matriz[linha][coluna] == 1:
                    altura = len(matriz) - linha
                    break
            if altura != 0:
                break
        return altura - 4

    def calc_holes(self):
        holes = 0

        for col in zip(*self.current_state()):
            i = 0
            while i < Board.HEIGHT and col[i] != 1:
                i += 1
            holes += len([x for x in col[i + 1 :] if x == 0])

        return holes

    def calc_bumpiness(self):
        total = 0

        min_ys = []
        for col in zip(*self.current_state()):
            i = 0
            while i < Board.HEIGHT and col[i] != 1:
                i += 1
            min_ys.append(i)
        for i in range(len(min_ys) - 1):
            total += abs(min_ys[i] - min_ys[i + 1])
        return total

    def calc_median_height(self):
        heights_sum = 0
        for col in zip(*self.matrix):
            for i in range(Board.HEIGHT - 1, -1, -1):
                if col[i] is not None:
                    heights_sum += i
        return float(heights_sum) / Board.WIDTH

    def calc_max_deviation_height(self):
        max_height = Board.HEIGHT - 1
        min_height = 0
        for col in zip(*self.matrix):
            for i in range(Board.HEIGHT - 1, -1, -1):
                if col[i] is not None:
                    if max_height is None or i > max_height:
                        max_height = i
                    if min_height is None or i < min_height:
                        min_height = i
        heights_delta = abs(max_height - min_height)
        return heights_delta

    def move_left(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.move_relative(Vec2(-1, 0))

        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            self.move_count_fall += 1

    def move_right(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.move_relative(Vec2(1, 0))

        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            self.move_count_fall += 1

    def swap(self):
        if self.did_swap_current_piece:
            return

        self.did_swap_current_piece = True
        self.did_turn_last_move = False
        self.did_turn_move_2_side_1_down = False

        if self.hold_piece is None:
            self.hold_piece = self.current.type
            self.current = self.queue.next()
            self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))
        else:
            temp = self.hold_piece
            self.hold_piece = self.current.type
            self.current = Tetromino.from_type(temp)
            self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

        self.fall_dt = 0
        self.fall_move_dt = 0

    def soft_drop(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()

        points = 0
        lines = 0
        if self.check_collision(tetromino_copy):
            lines, lock_points = self.lock_current_in_matrix()
            if lock_points is None:
                return lines, None
            points += lock_points
        else:
            self.did_turn_last_move = False
            self.did_turn_move_2_side_1_down = False
            self.current = tetromino_copy
            points += self.add_drop_score(ScoreType.SoftDrop, 1)

        return lines, points

    def hard_drop(self):
        height = self.ghost_current().origin.y - self.current.origin.y
        self.current = self.ghost_current()
        points = 0
        lines = 0
        lock_lines, lock_points = self.lock_current_in_matrix()
        if lock_lines is None:
            lines = 0
        if lock_points is None:
            return lines, None
        points += lock_points
        points += self.add_drop_score(ScoreType.HardDrop, height)
        return lines, points

    def check_collision(self, tetromino):
        for pos in tetromino.positions:
            if pos.x < 0 or pos.x >= Board.WIDTH:
                return True
            if pos.y >= Board.HEIGHT or (
                pos.y >= 0 and self.matrix[int(pos.y)][int(pos.x)] is not None
            ):
                return True
        return False

    def tick(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy

    def step(self, action):
        points = 0
        lines = 0
        reward = 0
        terminated = False

        if action == Action.SoftDrop:
            lock_lines, lock_points = self.soft_drop()
            if lock_points is None:
                terminated = True
            else:
                points += lock_points
        elif action == Action.HardDrop:
            lock_lines, lock_points = self.hard_drop()
            if lock_points is None:
                terminated = True
            else:
                points += lock_points
        else:
            if action == Action.RotateLeft:
                self.try_turn_current(Direction.Left)
            elif action == Action.RotateRight:
                self.try_turn_current(Direction.Right)
            elif action == Action.Switch:
                self.swap()
            elif action == Action.Left:
                self.move_left()
            elif action == Action.Right:
                self.move_right()

            tetromino_copy = copy.deepcopy(self.current)
            tetromino_copy.fall()
            if self.check_collision(tetromino_copy):
                lock_lines, lock_points = self.lock_current_in_matrix()
                if lock_points is None:
                    terminated = True
                else:
                    lines = lock_lines
                    points += lock_points

        reward = 1 + (lines**2) * Board.WIDTH

        # print(f"Reward = {reward} - {0.51*self.calcular_altura()} - {0.36*self.calc_holes()} - {0.18*self.calc_bumpiness()}")
        # reward -= 0.1 * self.calcular_altura()
        # reward -= 0.36 * self.calc_holes()
        # reward -= 0.18*self.calc_bumpiness()
        reward -= 0.2 * self.calc_max_deviation_height()

        if not terminated:
            self.tick()
        else:
            reward = -1000

        return self.current_state(), points, reward, terminated

    def update(self, dt):
        terminated = False
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()

        if self.check_collision(tetromino_copy):
            self.fall_dt += dt
            if (self.fall_dt > Board.LOCK_DELAY_MS) or (
                self.fall_move_dt > Board.MOVE_LOCK_DELAY_LIMIT
            ):
                lock_lines, lock_points = self.lock_current_in_matrix()
                if lock_points == None:
                    terminated = True
                    return terminated
        else:
            self.fall_dt = 0
            self.fall_move_dt = 0

        time = pygame.time.get_ticks()

        if (time - self.last_tick_ms_score) > 600:
            self.score_alpha -= 33
            self.last_tick_ms_score = time

        if (time - self.last_tick_ms) > Board.TICK_DT * (
            0.8 - ((self.level - 1) * 0.007)
        ) ** (self.level - 1):
            self.last_tick_ms = time
            self.tick()

        return terminated

    def lock_current_in_matrix(self):
        if self.check_collision(self.current):
            # print(f"Fim de jogo. Pontuação: {self.score}")
            return 0, None
        for pos in self.current.positions:
            self.matrix[int(pos.y)][int(pos.x)] = self.current.color

        lines, points = self.clear_rows()

        self.current = self.queue.next()
        self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()
        if self.check_collision(tetromino_copy):
            # print(f"Fim de jogo. Pontuação: {self.score}")
            return 0, None

        self.did_swap_current_piece = False
        self.did_turn_last_move = False
        self.did_turn_move_2_side_1_down = False
        return lines, points

    def clear_rows(self):
        new_matrix = list(filter(lambda row: not self.full_row(row), self.matrix))
        lines_cleared = Board.HEIGHT - len(new_matrix)
        if lines_cleared == 0:
            self.combo = 0
        points = self.add_score(
            lines_cleared, self.is_mini_t_spin(), self.is_full_t_spin(), False
        )
        if len(new_matrix) < Board.HEIGHT:
            new_matrix = [
                [None for _ in range(Board.WIDTH)]
                for _ in range(Board.HEIGHT - len(new_matrix))
            ] + new_matrix
        self.matrix = new_matrix
        return lines_cleared**2, points

    def full_row(self, row):
        for tile in row:
            if tile is None:
                return False
        return True

    def get_t_piece_front_minoes(self):
        if self.current.rotation == Rotation.Origin:
            return [
                self.matrix[self.current.origin.y - 1][self.current.origin.x - 1],
                self.matrix[self.current.origin.y - 1][self.current.origin.x + 1],
            ]
        elif self.current.rotation == Rotation.Left:
            return [
                self.matrix[self.current.origin.y - 1][self.current.origin.x + 1],
                self.matrix[self.current.origin.y + 1][self.current.origin.x + 1],
            ]
        elif self.current.rotation == Rotation.Right:
            return [
                self.matrix[self.current.origin.y - 1][self.current.origin.x - 1],
                self.matrix[self.current.origin.y + 1][self.current.origin.x - 1],
            ]
        else:
            return [
                self.matrix[self.current.origin.y + 1][self.current.origin.x - 1],
                self.matrix[self.current.origin.y + 1][self.current.origin.x + 1],
            ]

    def get_t_piece_back_minoes(self):
        if self.current.rotation == Rotation.Origin:
            if self.current.origin.y == (Board.HEIGHT - 1):
                return ["white", "white"]
            return [
                self.matrix[self.current.origin.y + 1][self.current.origin.x - 1],
                self.matrix[self.current.origin.y + 1][self.current.origin.x + 1],
            ]
        elif self.current.rotation == Rotation.Left:
            if self.current.origin.x == (Board.WIDTH - 1):
                return ["white", "white"]
            return [
                self.matrix[self.current.origin.y - 1][self.current.origin.x + 1],
                self.matrix[self.current.origin.y + 1][self.current.origin.x + 1],
            ]
        elif self.current.rotation == Rotation.Right:
            if self.current.origin.x == 0:
                return ["white", "white"]
            return [
                self.matrix[self.current.origin.y - 1][self.current.origin.x - 1],
                self.matrix[self.current.origin.y + 1][self.current.origin.x - 1],
            ]
        else:
            if self.current.origin.y == 0:
                return ["white", "white"]
            return [
                self.matrix[self.current.origin.y - 1][self.current.origin.x - 1],
                self.matrix[self.current.origin.y - 1][self.current.origin.x + 1],
            ]

    def qtd_t_piece_front_minoes(self):
        total = 0
        pieces = self.get_t_piece_front_minoes()
        if pieces[0] is not None:
            total += 1
        if pieces[1] is not None:
            total += 1
        return total

    def qtd_t_piece_back_minoes(self):
        total = 0
        pieces = self.get_t_piece_back_minoes()
        if pieces[0] is not None:
            total += 1
        if pieces[1] is not None:
            total += 1
        return total

    def is_full_t_spin(self):
        if self.current.type != TetrominoType.T or not self.did_turn_last_move:
            return False
        back_pieces = self.qtd_t_piece_back_minoes()
        front_pieces = self.qtd_t_piece_front_minoes()
        if back_pieces + front_pieces < 3:
            return False
        return front_pieces == 2 or self.did_turn_move_2_side_1_down

    def is_mini_t_spin(self):
        if self.current.type != TetrominoType.T or not self.did_turn_last_move:
            return False
        back_pieces = self.qtd_t_piece_back_minoes()
        front_pieces = self.qtd_t_piece_front_minoes()
        if back_pieces + front_pieces < 3:
            return False
        return back_pieces == 2

    def ghost_current(self):
        copy_tetromino = copy.deepcopy(self.current)
        while True:
            if self.check_collision(copy_tetromino):
                copy_tetromino.move_relative(Vec2(0, -1))
                return copy_tetromino
            copy_tetromino.move_relative(Vec2(0, 1))

    def add_score(self, qtd_lines, is_Tspin, is_mini_Tspin, b2b):
        points = 0
        if qtd_lines == 1 and not is_Tspin and not is_mini_Tspin:
            points += 100 * self.level
            self.last_score_type = "Single"
        elif qtd_lines == 2 and not is_Tspin and not is_mini_Tspin:
            points += 300 * self.level
            self.last_score_type = "Double"
        elif qtd_lines == 3 and not is_Tspin and not is_mini_Tspin:
            points += 500 * self.level
            self.last_score_type = "Triple"
        elif qtd_lines == 4 and not is_Tspin and not is_mini_Tspin:
            points += 800 * self.level
            self.last_score_type = "Tetris"
        elif qtd_lines == 0 and not is_Tspin and is_mini_Tspin:
            points += 100 * self.level
            self.last_score_type = "Mini T-Spin"
        elif qtd_lines == 0 and is_Tspin and not is_mini_Tspin:
            points += 400 * self.level
            self.last_score_type = "T-Spin"
        elif qtd_lines == 1 and not is_Tspin and is_mini_Tspin:
            points += 200 * self.level
            self.last_score_type = "Mini T-Spin Single"
        elif qtd_lines == 1 and is_Tspin and not is_mini_Tspin:
            points += 800 * self.level
            self.last_score_type = "T-Spin Single"
        elif qtd_lines == 1 and not is_Tspin and is_mini_Tspin:
            points += 400 * self.level
            self.last_score_type = "Mini T-Spin Double"
        elif qtd_lines == 2 and is_Tspin and not is_mini_Tspin:
            points += 1200 * self.level
            self.last_score_type = "T-Spin Double"
        elif qtd_lines == 3 and is_Tspin and not is_mini_Tspin:
            points += 1600 * self.level
            self.last_score_type = "T-Spin Triple"

        if self.combo >= 1:
            points += 50 * self.combo * self.level
            self.combo += 1
        else:
            self.combo = 1

        if b2b:
            points = int(1.5 * points)
        if points > 0:
            self.score_alpha = 300

        self.score += points
        self.score_level += points

        if self.score_level >= 1000:
            self.score_level = 0
            self.level += 1
        return points

    def current_state(self):
        state = []
        for row in self.matrix:
            state_row = []
            for item in row:
                if item is None:
                    state_row.append(0)
                else:
                    state_row.append(1)
            state.append(state_row)

        for pos in self.current.positions:
            if pos.y >= 0:
                state[int(pos.y)][int(pos.x)] = 2

        return state

    def display_current_state(self):
        state = self.current_state()
        for row in state:
            str_row = "".join(map(str, row))
            print(str_row)
        print()

    def add_drop_score(self, score_type, cells):
        sum = 0
        if score_type == ScoreType.SoftDrop:
            sum = cells
        elif ScoreType.HardDrop == ScoreType.HardDrop:
            sum = cells * 2

        self.score += sum
        self.score_level += sum

        if self.score_level >= 1000:
            self.score_level = 0
            self.level += 1
        return sum

    def get_game_score(self):
        return self.score
