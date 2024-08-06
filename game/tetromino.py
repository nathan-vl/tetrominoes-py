from dataclasses import dataclass
from enum import IntEnum
from typing import List
from game.direction import Direction
from game.rotation import Rotation
from game.utils import Vec2


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


class TetrominoType(IntEnum):
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
    rotation: Rotation = Rotation.Origin

    def get_wall_kick_tests(self, destRotation):
        if self.type == TetrominoType.I:
            return WALL_KICK_DATA_I_TETROMINO[self.rotation][destRotation]
        return WALL_KICK_DATA_COMMON[self.rotation][destRotation]

    @staticmethod
    def from_type(type):
        if type == TetrominoType.I:
            return Tetromino.__I()
        if type == TetrominoType.J:
            return Tetromino.__J()
        if type == TetrominoType.L:
            return Tetromino.__L()
        if type == TetrominoType.O:
            return Tetromino.__O()
        if type == TetrominoType.S:
            return Tetromino.__S()
        if type == TetrominoType.T:
            return Tetromino.__T()
        if type == TetrominoType.Z:
            return Tetromino.__Z()
        
    def turn(self, direction):
        if direction == Direction.Left:
            self.__turn_anticlockwise()
        else:
            self.__turn_clockwise()

    def __turn_clockwise(self):
        self.rotation = Rotation.rotate(self.rotation, Direction.Left)
        for i, pos in enumerate(self.positions):
            pos -= self.origin
            temp = pos.x
            pos.x = -pos.y
            pos.y = temp
            pos += self.origin

            self.positions[i] = pos

    def __turn_anticlockwise(self):
        self.rotation = Rotation.rotate(self.rotation, Direction.Right)
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
    def __I():
        return Tetromino(
            Vec2(1.5, 1.5),
            [Vec2(0, 1), Vec2(1, 1), Vec2(2, 1), Vec2(3, 1)],
            "cyan",
            TetrominoType.I,
        )

    @staticmethod
    def __J():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(0, 0), Vec2(0, 1), Vec2(1, 1), Vec2(2, 1)],
            "blue",
            TetrominoType.J,
        )

    @staticmethod
    def __L():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(2, 0), Vec2(0, 1), Vec2(1, 1), Vec2(2, 1)],
            "orange",
            TetrominoType.L,
        )

    @staticmethod
    def __O():
        return Tetromino(
            Vec2(1.5, 0.5),
            [Vec2(1, 0), Vec2(2, 0), Vec2(1, 1), Vec2(2, 1)],
            "yellow",
            TetrominoType.O,
        )

    @staticmethod
    def __S():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(1, 0), Vec2(2, 0), Vec2(0, 1), Vec2(1, 1)],
            "lime",
            TetrominoType.S,
        )

    @staticmethod
    def __T():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(0, 1), Vec2(1, 0), Vec2(1, 1), Vec2(2, 1)],
            "purple",
            TetrominoType.T,
        )

    @staticmethod
    def __Z():
        return Tetromino(
            Vec2(1, 1),
            [Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(2, 1)],
            "red",
            TetrominoType.Z,
        )
