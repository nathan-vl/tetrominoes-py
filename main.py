import pygame
from board import Board
from board_view import BoardView

FPS = 60
TILE_SIZE = 20


def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
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
                    board.turn_anticlockwise()
                    board.fall_move_dt += 1
                    board.fall_dt = 0
                if event.key == pygame.K_x:
                    board.turn_clockwise()
                    board.fall_move_dt += 1
                    board.fall_dt = 0
                if event.key == pygame.K_LEFT:
                    board.move_left()
                    board.fall_move_dt += 1
                if event.key == pygame.K_RIGHT:
                    board.move_right()
                    board.fall_move_dt += 1
                if event.key == pygame.K_DOWN:
                    board.soft_drop()

        screen.fill("grey")
        screen.blit(BoardView.render(board), (100, 100))
        pygame.display.flip()

        board.update(dt)

        dt = clock.tick(FPS)

    pygame.quit()


main()
