import sys
import pygame
import numpy as np
from consts import WINDOW_SIZE, BOARD_SIZE
from helper import from_screen, to_screen
from game import draw_board, draw_hover, draw_ui
from goBan import GoGame


def main_menu(screen, mcts):
    bg = pygame.image.load("assets\goBG.png")
    while True:
        screen.blit(bg, (0, 0))
        
        mx, my = pygame.mouse.get_pos()
        startButton = pygame.Rect(50, 100, 200, 50)
        quitButton = pygame.Rect(50, 200, 200, 50)
        if startButton.collidepoint((mx, my)):
            gameScreen(screen, mcts)
        if quitButton.collidepoint((mx, my)):
            pass
        # pygame.draw.rect(screen, (255, 0, 0), startButton)
        # pygame.draw.rect(screen, (255, 0, 0), quitButton)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        pygame.display.update()


def gameScreen(screen, mcts):
    message = ""
    hover = None
    current_player = 1
    passes = 0
    game = GoGame(BOARD_SIZE)
    board = game.getInitBoard() 

    while True:
        # mouse position
        mx, my = pygame.mouse.get_pos()
        # for hover effect
        hover = from_screen(mx, my)
        # valids = game.getValidMoves(board, current_player)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN and current_player == 1:            
                mx, my = event.pos
                if my >= WINDOW_SIZE:
                    # UI clicked
                    pass_rect, reset_rect = draw_ui(screen, hover, message)
                    if pass_rect.collidepoint(mx, my):
                        passes += 1
                        message = f"{'Black' if current_player==1 else 'White'} passed."
                        if passes >= 2:
                            game_over = True
                            message = "Both players passed. Game over (counting not implemented)."
                        continue
                    if reset_rect.collidepoint(mx, my):
                        board = game.getInitBoard()
                        current_player = 1
                        passes = 0
                        game_over = False
                        message = "Board reset."
                        continue
                # Board click
                pos = from_screen(mx, my)
                if pos is not None:
                    i, j = pos
                    action = i * game.n + j
                    board, current_player = game.getNextState(board, current_player, action)
                    game.display(board)

        # AI play
        if current_player == -1:
            neutral_state = game.getCanonicalForm(board, current_player)
            mcts_probs = mcts.search(neutral_state, current_player)
            action = np.argmax(mcts_probs)
            board, current_player = game.getNextState(board, current_player, action)
            game.display(board)

        # render game here
        draw_board(screen, board)

        if hover:
            hi, hj = hover
            if board[hi][hj] == 0:
                draw_hover(screen, hover, current_player)

        # draw_board(screen)
        pass_rect, reset_rect = draw_ui(screen, hover, message)


        pygame.display.flip()


