import pygame


class TetrominoView:
    @staticmethod
    def render(tetromino, surface, tile_size):
        for pos in tetromino.positions:
            pygame.draw.rect(
                surface,
                tetromino.color,
                pygame.Rect(
                    pos.x * tile_size,
                    pos.y * tile_size,
                    tile_size,
                    tile_size,
                ),
            )
