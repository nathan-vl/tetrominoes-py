import pygame

from tetromino import Tetromino
from tetromino_view import TetrominoView

class HoldView:
    @staticmethod
    def render(hold_piece, tile_size):
        surface = pygame.Surface((4 * tile_size, 4 * tile_size))
        if hold_piece is not None:
            TetrominoView.render(Tetromino.fromType(hold_piece), surface, tile_size)
        return surface
