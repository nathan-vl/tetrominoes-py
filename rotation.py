from enum import IntEnum
from direction import Direction


class Rotation(IntEnum):
    Origin = 0
    Right = 1
    Double = 2
    Left = 3

    @staticmethod
    def rotate(rotation, direction):
        if direction == Direction.Left:
            return (rotation + 3) % 4
        else:
            return (rotation + 1) % 4
