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
        # Não está sendo enviado a ação Hard Drop
        return random.choice(range(0, 6))

    @staticmethod
    def display(action):
        if action == 0:
            print("L")
        elif action == 1:
            print("R")
        elif action == 2:
            print("RL")
        elif action == 3:
            print("RR")
        elif action == 4:
            print("S")
        elif action == 5:
            print("D")
        elif action == 6:
            print("H")
