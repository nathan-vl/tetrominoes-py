import pygame
from board import Board
from tetromino_view import TetrominoView


class BoardView:
    @staticmethod
    def surface(board):
        surface = pygame.Surface(
            (
                Board.WIDTH * Board.TILE_SIZE,
                Board.HEIGHT * Board.TILE_SIZE,
            ),
            pygame.SRCALPHA,
        )

        surface.fill("black")
        BoardView.__render_pieces(board, surface)
        BoardView.__render_ghost(surface, board)
        TetrominoView.render(board.current, surface, Board.TILE_SIZE)

        BoardView.__render_vertical_gridlines(surface)
        BoardView.__render_horizontal_gridlines(surface)

        return surface

    @staticmethod
    def __render_pieces(board, surface):
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
    def __render_horizontal_gridlines(surface):
        for i in range(Board.HEIGHT):
            pygame.draw.line(
                surface,
                "gray45",
                (0, i * Board.TILE_SIZE),
                (Board.WIDTH * Board.TILE_SIZE, i * Board.TILE_SIZE),
            )

    @staticmethod
    def __render_vertical_gridlines(surface):
        for i in range(Board.WIDTH):
            pygame.draw.line(
                surface,
                "gray45",
                (i * Board.TILE_SIZE, 0),
                (i * Board.TILE_SIZE, Board.HEIGHT * Board.TILE_SIZE),
            )

    @staticmethod
    def __render_ghost(surface, board):
        ghost = board.ghost_current()
        color = pygame.Color(ghost.color)
        color.a = 150
        ghost.color = color
        TetrominoView.render(ghost, surface, Board.TILE_SIZE)

    @staticmethod
    def render_score(board):
        surface = pygame.Surface(
            (
                100,
                100,
            ),
            pygame.SRCALPHA,
        )
        surface.fill("black")
        BoardView.__render_text(surface, board)
        return surface

    @staticmethod
    def __render_text(surface, board):
        pygame.font.init()
        my_font = pygame.font.SysFont("microsofttaile", 25)
        score_points = my_font.render(str(int(board.score)), False, (255, 255, 255))
        text_score = my_font.render(str(board.last_score_type), False, (255, 255, 255))
        text_score.set_alpha(board.score_alpha)
        surface.blit(score_points, (0, 0))
        surface.blit(text_score, (0, 50))
