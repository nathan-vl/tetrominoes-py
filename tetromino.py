from dataclasses import dataclass
from enum import Enum
from typing import List
from utils import Vec2


class TetrominoType(Enum):
    I = 0
    J = 1
    L = 2
    O = 3
    S = 4
    T = 5
    Z = 6


@dataclass
class Tetromino:
    origin: Vec2
    positions: List[Vec2]
    color: str
    type: TetrominoType

    def turn_clockwise(self):
        for i, pos in enumerate(self.positions):
            pos -= self.origin
            temp = pos.x
            pos.x = -pos.y
            pos.y = temp
            pos += self.origin

            self.positions[i] = pos

    def turn_anticlockwise(self):
        for i, pos in enumerate(self.positions):
            pos -= self.origin
            temp = pos.x
            pos.x = pos.y
            pos.y = -temp
            pos += self.origin

            self.positions[i] = pos

    def fall(self):
        self.origin.y += 1
        for i in range(len(self.positions)):
            self.positions[i].y += 1
    
    def move(self, pos):
        distance = pos - self.origin
        if self.type == TetrominoType.O or self.type == TetrominoType.I:
            distance += Vec2(0.5, 0.5)
        self.move_relative(distance)

    def move_relative(self, pos):
        self.origin += pos
        for i in range(len(self.positions)):
            self.positions[i] += pos
    
    def get_pieces_height(self):
        return [pos.x for pos in self.positions]

    def move_inside(self, width):
        heights = self.get_pieces_height()
        min_x = min(heights)
        max_x = max(heights)

        if min_x < 0:
            self.move_relative(Vec2(-min_x, 0))
        if max_x >= width:
            self.move_relative(Vec2(width - max_x - 1, 0))

    @staticmethod
    def I():
        return Tetromino(
            Vec2(1.5, 1.5),
            [Vec2(0, 1), Vec2(1, 1), Vec2(2, 1), Vec2(3, 1)],
            "cyan",
            TetrominoType.I,
        )

    @staticmethod
    def J():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(0, 0), Vec2(0, 1), Vec2(1, 1), Vec2(2, 1)],
            "blue",
            TetrominoType.J,
        )

    @staticmethod
    def L():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(2, 0), Vec2(0, 1), Vec2(1, 1), Vec2(2, 1)],
            "orange",
            TetrominoType.L,
        )

    @staticmethod
    def O():
        return Tetromino(
            Vec2(1.5, 0.5),
            [Vec2(1, 0), Vec2(2, 0), Vec2(1, 1), Vec2(2, 1)],
            "yellow",
            TetrominoType.O,
        )

    @staticmethod
    def S():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(1, 0), Vec2(2, 0), Vec2(0, 1), Vec2(1, 1)],
            "lime",
            TetrominoType.S,
        )

    @staticmethod
    def T():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(0, 1), Vec2(1, 0), Vec2(1, 1), Vec2(2, 1)],
            "purple",
            TetrominoType.T,
        )

    @staticmethod
    def Z():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(2, 1)],
            "red",
            TetrominoType.Z,
        )
