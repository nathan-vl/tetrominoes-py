import pygame
from game.board import Board
from game.board_view import BoardView
from game.direction import Direction
from game.hold_view import HoldView
from game.queue_view import QueueView

FPS = 60
TILE_SIZE = 20


def main():
    pygame.init()
    screen = pygame.display.set_mode((410, 410))
    clock = pygame.time.Clock()
    dt = 0

    board = Board()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    board.try_turn_current(Direction.Left)
                    board.fall_move_dt += 1
                    board.fall_dt = 0
                if event.key == pygame.K_UP:
                    board.try_turn_current(Direction.Right)
                    board.fall_move_dt += 1
                    board.fall_dt = 0
                if event.key == pygame.K_c:
                    board.swap()
                if event.key == pygame.K_LEFT:
                    board.move_left()
                    board.fall_move_dt += 1
                if event.key == pygame.K_RIGHT:
                    board.move_right()
                    board.fall_move_dt += 1
                if event.key == pygame.K_DOWN:
                    board.soft_drop()
                if event.key == pygame.K_SPACE:
                    board.hard_drop()

        screen.fill("grey")

        board_surface = BoardView.surface(board)
        queue_surface = QueueView.surface(board.queue, TILE_SIZE)
        hold_surface = HoldView.surface(board.hold_piece, TILE_SIZE)
        score_surface = BoardView.render_score(board)

        screen.blit(board_surface, (110, 5))
        screen.blit(
            hold_surface, (100 + Board.TILE_SIZE + board_surface.get_width(), 5)
        )
        screen.blit(
            queue_surface,
            (100 + Board.TILE_SIZE + board_surface.get_width(), TILE_SIZE * 5 + 5),
        )
        screen.blit(score_surface, (5, 5))
        pygame.display.flip()

        result = board.update(dt)

        if result:
            break 
        
        dt = clock.tick(FPS)

    pygame.quit()


main()
