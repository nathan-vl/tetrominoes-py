import pygame

from tetromino import Tetromino
from tetromino_view import TetrominoView
from utils import Vec2


class HoldView:
    @staticmethod
    def surface(hold_piece, tile_size):
        surface = pygame.Surface((4 * tile_size, 4 * tile_size), pygame.SRCALPHA)
        surface.fill("black")
        if hold_piece is not None:
            tetromino = Tetromino.from_type(hold_piece)
            tetromino.move_relative(Vec2(0, 1))
            TetrominoView.render(tetromino, surface, tile_size)
        return surface
