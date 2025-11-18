import pygame
from GameState import GameState
from consts import WIDTH, HEIGHT, BOARD_SIZE
from screens import main_menu
from goBan import GoGame
from MCTS import MCTS
from GoNet import GoNet

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_state = GameState.TITLE

    game = GoGame(BOARD_SIZE)
    board = game.getInitBoard() 
    actionSize = game.getActionSize()
    args = {
        'C': 1.41,
        'num_searches': 10
    }
    model = GoNet(BOARD_SIZE, actionSize)
    mcts = MCTS(game, args, model)
    main_menu(screen, mcts)
    # while True:
    #     if game_state == GameState.TITLE:
    #         # game_state = title_screen(screen)
    #         pass
    #     if game_state == GameState.NEWGAME:
    #         game_state = game(screen, game, board. mcts)
    #     if game_state == GameState.QUIT:
    #         pygame.quit()
    #         return


if __name__ == "__main__":
    main()