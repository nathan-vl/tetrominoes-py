from enum import IntEnum
import random


class Action(IntEnum):
    Left = 5
    Right = 6
    RotateLeft = 2
    RotateRight = 3
    Switch = 4
    SoftDrop = 0
    HardDrop = 1

    @staticmethod
    def sample():
        return random.choice(range(0, 7))
