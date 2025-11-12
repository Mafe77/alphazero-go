import pygame
from GameState import GameState
from consts import WIDTH, HEIGHT, BOARD_SIZE
from screens import main_menu
from goBan import GoGame
from MCTS import MCTS

def main():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_state = GameState.TITLE

    game = GoGame(BOARD_SIZE)
    board = game.getInitBoard() 
    args = {
        'C': 1.41,
        'num_searches': 10
    }

    mcts = MCTS(game, args)
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