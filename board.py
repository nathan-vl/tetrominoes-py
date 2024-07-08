import copy
import random
from tetromino import Tetromino
from utils import Vec2

import pygame


class Board:
    TILE_SIZE = 20

    WIDTH = 10
    HEIGHT = 20

    TICK_DT = 300

    LOCK_DELAY_MS = 500
    MOVE_LOCK_DELAY_LIMIT = 15

    def __init__(self) -> None:
        self.matrix = [[None for _ in range(Board.WIDTH)] for _ in range(Board.HEIGHT)]
        self.tetrominoes_stack = Board.new_tetrominoes()

        self.current = self.tetrominoes_stack.pop()
        self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

        self.fall_dt = 0
        self.move_count_fall = 0
        self.fall_move_dt = 0
        self.last_tick_ms = 0

    def new_tetrominoes():
        TETROMINOES = [
            Tetromino.I(),
            Tetromino.J(),
            Tetromino.L(),
            Tetromino.O(),
            Tetromino.S(),
            Tetromino.T(),
            Tetromino.Z(),
        ]
        return random.sample(TETROMINOES, k=len(TETROMINOES))

    def turn_anticlockwise(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.turn_anticlockwise()

        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(1, 0))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(0, -1))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(-1, 3))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(1, 0))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy

    def turn_clockwise(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.turn_clockwise()

        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(-1, 0))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(0, -1))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(1, 3))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy
            return

        tetromino_copy.move_relative(Vec2(-1, 0))
        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy

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

    def soft_drop(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()

        if self.check_collision(tetromino_copy):
            self.lock_current_in_matrix()
        else:
            self.current = tetromino_copy

    def check_collision(self, tetromino):
        for pos in tetromino.positions:
            if pos.x < 0 or pos.x >= Board.WIDTH:
                return True
            if pos.y >= Board.HEIGHT or (pos.y >= 0 and self.matrix[int(pos.y)][int(pos.x)] is not None):
                return True
        return False

    def tick(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()

        if not self.check_collision(tetromino_copy):
            self.current = tetromino_copy

    def update(self, dt):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()

        if self.check_collision(tetromino_copy):
            self.fall_dt += dt
            if (self.fall_dt > Board.LOCK_DELAY_MS) or (
                self.fall_move_dt > Board.MOVE_LOCK_DELAY_LIMIT
            ):
                self.lock_current_in_matrix()
                self.clear_rows()
        else:
            self.fall_dt = 0
            self.fall_move_dt = 0

        time = pygame.time.get_ticks()
        if (time - self.last_tick_ms) > Board.TICK_DT:
            self.last_tick_ms = time
            self.tick()

    def lock_current_in_matrix(self):
        for pos in self.current.positions:
            self.matrix[int(pos.y)][int(pos.x)] = self.current.color

        if len(self.tetrominoes_stack) == 0:
            self.tetrominoes_stack = Board.new_tetrominoes()
        self.current = self.tetrominoes_stack.pop()
        self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

    def clear_rows(self):
        new_matrix = list(filter(lambda row: not self.full_row(row), self.matrix))
        if len(new_matrix) < Board.HEIGHT:
            new_matrix = [
                [None for _ in range(Board.WIDTH)]
                for _ in range(Board.HEIGHT - len(new_matrix))
            ] + new_matrix
        self.matrix = new_matrix

    def full_row(self, row):
        for tile in row:
            if tile is None:
                return False
        return True
