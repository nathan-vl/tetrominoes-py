from typing import List

from tetromino import Tetromino
from tetromino_view import TetrominoView
from utils import Vec2

import pygame


class QueueView:
    TETROMINO_MAX_OCCUPY_HEIGHT = 4

    @staticmethod
    def render(queue: List[Tetromino], tile_size):
        total_width = tile_size * QueueView.TETROMINO_MAX_OCCUPY_HEIGHT
        total_height = queue.VISIBLE_QUEUE_COUNT * tile_size * QueueView.TETROMINO_MAX_OCCUPY_HEIGHT

        surface = pygame.Surface((total_width, total_height))
        surface.fill("black")

        pieces = queue.current_queue()
        for i in range(queue.VISIBLE_QUEUE_COUNT):
            origin_y = i * QueueView.TETROMINO_MAX_OCCUPY_HEIGHT

            piece = Tetromino.fromType(pieces[i]) 
            piece.move_relative(Vec2(0, origin_y))

            TetrominoView.render(piece, surface, tile_size)

        return surface
