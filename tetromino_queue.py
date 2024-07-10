import random
from tetromino import Tetromino, TetrominoType


class TetrominoQueue:
    TETROMINOES = [
        TetrominoType.I,
        TetrominoType.J,
        TetrominoType.L,
        TetrominoType.O,
        TetrominoType.S,
        TetrominoType.T,
        TetrominoType.Z,
    ]
    VISIBLE_QUEUE_COUNT = 5

    def __init__(self):
        self.queue = TetrominoQueue.__bag()

    def next(self):
        piece = self.queue.pop(0)
        if len(self.queue) < TetrominoQueue.VISIBLE_QUEUE_COUNT:
            self.queue.extend(TetrominoQueue.__bag())
        return Tetromino.from_type(piece)

    def current_queue(self):
        queue = []
        for i in range(TetrominoQueue.VISIBLE_QUEUE_COUNT):
            queue.append(self.queue[i])
        return queue

    @staticmethod
    def __bag():
        return random.sample(
            TetrominoQueue.TETROMINOES, k=len(TetrominoQueue.TETROMINOES)
        )
