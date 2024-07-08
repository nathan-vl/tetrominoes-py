import pygame
from board import Board

FPS = 60
TILE_SIZE = 20
UPDATE_DT = 300


def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    dt = 0

    last_update_time = 0

    board = Board()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    board.turn_left()
                    board.fall_move_dt += 1
                    board.fall_dt = 0
                if event.key == pygame.K_x:
                    board.turn_right()
                    board.fall_move_dt += 1
                    board.fall_dt = 0
                if event.key == pygame.K_LEFT:
                    board.move_left()
                    board.fall_move_dt += 1
                if event.key == pygame.K_RIGHT:
                    board.move_right()
                    board.fall_move_dt += 1

        screen.fill("grey")
        screen.blit(board.render(), (100, 100))
        pygame.display.flip()

        board.update(dt)

        time = pygame.time.get_ticks()
        if (time - last_update_time) > UPDATE_DT:
            last_update_time = time
            board.tick()

        dt = clock.tick(FPS)

    pygame.quit()


main()
