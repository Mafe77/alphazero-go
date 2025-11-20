import sys
import pygame
import numpy as np
from consts import WINDOW_SIZE, BOARD_SIZE
from helper import from_screen, to_screen
from game import draw_board, draw_hover, draw_ui
from goBan import GoGame


def main_menu(screen, mcts, game, board):
    bg = pygame.image.load("assets/goBG.png")
    while True:
        screen.blit(bg, (0, 0))
        
        mx, my = pygame.mouse.get_pos()
        startButton = pygame.Rect(50, 100, 200, 50)
        quitButton = pygame.Rect(50, 200, 200, 50)
        if startButton.collidepoint((mx, my)):
            gameScreen(screen, mcts, game, board)
        if quitButton.collidepoint((mx, my)):
            pass
        pygame.draw.rect(screen, (255, 0, 0), startButton)
        pygame.draw.rect(screen, (255, 0, 0), quitButton)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        pygame.display.update()


def gameScreen(screen, mcts, game, board):
    running = True
    current_player = 1
    game_over = False
    passes = 0
    ai_processing = False
    handle_click = False

    board = game.getInitBoard()
    message = ""

    while running:
        mx, my = pygame.mouse.get_pos()
        hover = from_screen(mx, my)

        # -------------------------
        # EVENT HANDLING
        # -------------------------
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
                break

            if current_player == 1 and not game_over:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    handle_click = True
                    print("CLICK")

        # -------------------------
        # HUMAN MOVE
        # -------------------------
            if handle_click:
                handle_click = False   # avoid repeats

                pos = from_screen(mx, my)
                if pos:
                    i, j = pos
                    action = i * game.n + j

                    valids = game.getValidMoves(board, current_player)
                    if valids[action] == 1:
                        board, current_player = game.getNextState(board, 1, action)
                        print("CLICK2")
                        current_player = -1
                        passes = 0
                    else:
                        message = "Invalid move."

        # -------------------------
        # AI MOVE
        # -------------------------
            if current_player == -1 and not game_over and not ai_processing:
                ai_processing = True

                valids = game.getValidMoves(board, current_player)
                if valids.sum() == 0:
                    passes += 1
                    message = "White passed."

                    if passes >= 2:
                        game_over = True
                    else:
                        current_player = 1
                else:
                    neutral = game.getCanonicalForm(board, current_player)
                    mcts_probs = mcts.search(neutral)
                    action = np.argmax(mcts_probs)

                    board, current_player = game.getNextState(board, current_player, action)
                    passes = 0

                ai_processing = False

        # DRAW
        draw_board(screen, board)
        draw_ui(screen, hover, message)
        pygame.display.flip()

