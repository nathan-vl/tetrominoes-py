import copy
from enum import IntEnum
from utils import Vec2
from tetromino import Rotation, Tetromino, TetrominoType
from tetromino_queue import TetrominoQueue

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


WALL_KICK_DATA_COMMON = {
    Rotation.Origin: {
        Rotation.Right: [
            Vec2(0, 0),
            Vec2(-1, 0),
            Vec2(-1, -1),
            Vec2(0, 2),
            Vec2(-1, 2),
        ],
        Rotation.Left: [
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(1, -1),
            Vec2(0, 2),
            Vec2(1, 2),
        ],
    },
    Rotation.Right: {
        Rotation.Origin: [
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(1, 1),
            Vec2(0, -2),
            Vec2(1, -2),
        ],
        Rotation.Double: [
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(1, 1),
            Vec2(0, -2),
            Vec2(1, -2),
        ],
    },
    Rotation.Double: {
        Rotation.Right: [
            Vec2(0, 0),
            Vec2(-1, 0),
            Vec2(-1, -1),
            Vec2(0, 2),
            Vec2(-1, 2),
        ],
        Rotation.Left: [
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(1, -1),
            Vec2(0, 2),
            Vec2(1, 2),
        ],
    },
    Rotation.Left: {
        Rotation.Double: [
            Vec2(0, 0),
            Vec2(-1, 0),
            Vec2(-1, 1),
            Vec2(0, -2),
            Vec2(-1, -2),
        ],
        Rotation.Origin: [
            Vec2(0, 0),
            Vec2(-1, 0),
            Vec2(-1, 1),
            Vec2(0, -2),
            Vec2(-1, -2),
        ],
    },
}

WALL_KICK_DATA_I_TETROMINO = {
    Rotation.Origin: {
        Rotation.Right: [
            Vec2(0, 0),
            Vec2(-2, 0),
            Vec2(1, 0),
            Vec2(-2, 1),
            Vec2(1, -2),
        ],
        Rotation.Left: [
            Vec2(0, 0),
            Vec2(-1, 0),
            Vec2(2, 0),
            Vec2(-1, -2),
            Vec2(2, 1),
        ],
    },
    Rotation.Right: {
        Rotation.Origin: [
            Vec2(0, 0),
            Vec2(2, 0),
            Vec2(-1, 0),
            Vec2(2, -1),
            Vec2(-1, 2),
        ],
        Rotation.Double: [
            Vec2(0, 0),
            Vec2(-1, 0),
            Vec2(2, 0),
            Vec2(-1, -2),
            Vec2(2, 1),
        ],
    },
    Rotation.Double: {
        Rotation.Right: [
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(-2, 0),
            Vec2(1, 2),
            Vec2(-2, -1),
        ],
        Rotation.Left: [
            Vec2(0, 0),
            Vec2(2, 0),
            Vec2(-1, 0),
            Vec2(2, -1),
            Vec2(-1, 2),
        ],
    },
    Rotation.Left: {
        Rotation.Double: [
            Vec2(0, 0),
            Vec2(-2, 0),
            Vec2(1, 0),
            Vec2(-2, 1),
            Vec2(1, -2),
        ],
        Rotation.Origin: [
            Vec2(0, 0),
            Vec2(1, 0),
            Vec2(-2, 0),
            Vec2(1, 2),
            Vec2(-2, -1),
        ],
    },
}


class Board:
    TILE_SIZE = 20

    WIDTH = 10
    HEIGHT = 20

    TICK_DT = 300

    LOCK_DELAY_MS = 500
    MOVE_LOCK_DELAY_LIMIT = 15

    def __init__(self):
        self.matrix = [[None for _ in range(Board.WIDTH)] for _ in range(Board.HEIGHT)]
        self.queue = TetrominoQueue()

        self.hold_piece = None
        self.current = self.queue.next()
        self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

        self.did_swap_current_piece = False
        self.fall_dt = 0
        self.move_count_fall = 0
        self.fall_move_dt = 0
        self.last_tick_ms = 0
        self.level = 1
        self.score = 0
        self.last_score_type = ""
        self.score_alpha = 0

    def turn_anticlockwise(self):
        kick_tests = []
        if self.current.type == TetrominoType.I:
            kick_tests = WALL_KICK_DATA_I_TETROMINO[self.current.rotation][
                Rotation.anticlockwise(self.current.rotation)
            ]
        else:
            kick_tests = WALL_KICK_DATA_COMMON[self.current.rotation][
                Rotation.anticlockwise(self.current.rotation)
            ]

        for test in kick_tests:
            tetromino_copy = copy.deepcopy(self.current)
            tetromino_copy.turn_anticlockwise()

            tetromino_copy.move_relative(test)
            if not self.check_collision(tetromino_copy):
                self.current = tetromino_copy
                # TODO: Reset counters
                return

    def turn_clockwise(self):
        kick_tests = []
        if self.current.type == TetrominoType.I:
            kick_tests = WALL_KICK_DATA_I_TETROMINO[self.current.rotation][
                Rotation.clockwise(self.current.rotation)
            ]
        else:
            kick_tests = WALL_KICK_DATA_COMMON[self.current.rotation][
                Rotation.clockwise(self.current.rotation)
            ]

        for test in kick_tests:
            tetromino_copy = copy.deepcopy(self.current)
            tetromino_copy.turn_clockwise()

            tetromino_copy.move_relative(test)
            if not self.check_collision(tetromino_copy):
                self.current = tetromino_copy
                # TODO: Reset counters
                return

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

        if self.hold_piece is None:
            self.hold_piece = self.current.type
            self.current = self.queue.next()
            self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))
        else:
            temp = self.hold_piece
            self.hold_piece = self.current.type
            self.current = Tetromino.fromType(temp)
            self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))

        self.fall_dt = 0
        self.fall_move_dt = 0

    def soft_drop(self):
        tetromino_copy = copy.deepcopy(self.current)
        tetromino_copy.fall()

        if self.check_collision(tetromino_copy):
            self.lock_current_in_matrix()
        else:
            self.current = tetromino_copy

    def hard_drop(self):
        self.current = self.ghost_current()
        self.lock_current_in_matrix()

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
        self.score_alpha -= 100
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
        if self.check_collision(self.current):
            print(f"Fim de jogo. Pontuação: {self.score}")
            exit(0)
        for pos in self.current.positions:
            self.matrix[int(pos.y)][int(pos.x)] = self.current.color

        self.current = self.queue.next()
        self.current.move(Vec2((Board.WIDTH - 1) // 2, -1))
        self.did_swap_current_piece = False

    def clear_rows(self):
        new_matrix = list(filter(lambda row: not self.full_row(row), self.matrix))
        lines_cleared = Board.HEIGHT - len(new_matrix)
        self.add_score(lines_cleared - 1, False)
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

    def ghost_current(self):
        copy_tetromino = copy.deepcopy(self.current)
        while True:
            if self.check_collision(copy_tetromino):
                copy_tetromino.move_relative(Vec2(0, -1))
                return copy_tetromino
            copy_tetromino.move_relative(Vec2(0, 1))

    def add_score(self, score_type, b2b):
        points = 0
        if score_type == ScoreType.Single:
            points += 100 * self.level
            self.last_score_type = "Single"
            self.score_alpha = 300
        elif score_type == ScoreType.Double:
            points += 300 * self.level
            self.last_score_type = "Double"
            self.score_alpha = 300
        elif score_type == ScoreType.Triple:
            points += 500 * self.level
            self.last_score_type = "Triple"
            self.score_alpha = 300
        elif score_type == ScoreType.Tetris:
            points += 800 * self.level
            self.last_score_type = "Tetris"
            self.score_alpha = 300
        elif score_type == ScoreType.MiniTSpin:
            points += 100 * self.level
        elif score_type == ScoreType.TSpinNoline:
            points += 400 * self.level
        elif score_type == ScoreType.MiniTSpinSingle:
            points += 200 * self.level
        elif score_type == ScoreType.TSpinSingle:
            points += 800 * self.level
        elif score_type == ScoreType.MiniTSpinDouble:
            points += 400 * self.level
        elif score_type == ScoreType.TSpinDouble:
            points += 1200 * self.level
        elif score_type == ScoreType.TSpinTriple:
            points += 1600 * self.level
        elif score_type == ScoreType.BackToBack:
            points += 800 * self.level

        if b2b:
            points = int(1.5 * points)

        self.score += points

    def add_drop_score(self, score_type, cells):
        sum = 0
        match score_type:
            case ScoreType.SoftDrop:
                sum = cells
            case ScoreType.HardDrop:
                sum = cells * 2
        self.score += sum
