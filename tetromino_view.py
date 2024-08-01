import pygame


class TetrominoView:
    @staticmethod
    def __tile_surface(tile_size, color):
        surface = pos_surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
        pygame.draw.rect(
            pos_surface,
            color,
            (0, 0, tile_size, tile_size),
        )
        return surface

    @staticmethod
    def render(tetromino, surface, tile_size):
        for pos in tetromino.positions:
            target_rect = pygame.Rect(
                pos.x * tile_size,
                pos.y * tile_size,
                tile_size,
                tile_size,
            )
            surface.blit(TetrominoView.__tile_surface(tile_size, tetromino.color), target_rect)
