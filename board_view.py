import pygame
from board import Board
from tetromino_view import TetrominoView

class BoardView:
    @staticmethod
    def render(board):
        surface = pygame.Surface(
            (
                Board.WIDTH * Board.TILE_SIZE,
                Board.HEIGHT * Board.TILE_SIZE,
            )
        )

        surface.fill("black")
        BoardView.render_pieces(board, surface)
        TetrominoView.render(board.current, surface, Board.TILE_SIZE)

        BoardView.render_vertical_gridlines(surface)
        BoardView.render_horizontal_gridlines(surface)

        return surface

    @staticmethod
    def render_pieces(board, surface):
        for y, row in enumerate(board.matrix):
            for x, tile in enumerate(row):
                if tile is not None:
                    pygame.draw.rect(
                        surface,
                        tile,
                        pygame.Rect(
                            x * Board.TILE_SIZE,
                            y * Board.TILE_SIZE,
                            Board.TILE_SIZE,
                            Board.TILE_SIZE,
                        ),
                    )

    @staticmethod
    def render_horizontal_gridlines(surface):
        for i in range(Board.HEIGHT):
            pygame.draw.line(
                surface,
                "gray45",
                (0, i * Board.TILE_SIZE),
                (Board.WIDTH * Board.TILE_SIZE, i * Board.TILE_SIZE),
            )

    @staticmethod
    def render_vertical_gridlines(surface):
        for i in range(Board.WIDTH):
            pygame.draw.line(
                surface,
                "gray45",
                (i * Board.TILE_SIZE, 0),
                (i * Board.TILE_SIZE, Board.HEIGHT * Board.TILE_SIZE),
            )
