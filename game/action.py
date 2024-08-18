from enum import IntEnum
import random


class Action(IntEnum):
    Left = 0
    Right = 1
    RotateLeft = 2
    RotateRight = 3
    Switch = 4
    SoftDrop = 5
    HardDrop = 6

    @staticmethod
    def sample():
        return random.choice(range(0, 5))
