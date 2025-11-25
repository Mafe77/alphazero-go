import pygame
import torch
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = GoGame(BOARD_SIZE)
    board = game.getInitBoard()
    actionSize = game.getActionSize()
    args = {
        'C': 1.41,
        'num_searches': 1000
    }
    model = GoNet(BOARD_SIZE, actionSize)
    model.load_state_dict(torch.load("model_2.pt", map_location=device))
    model.eval()
    mcts = MCTS(game, args, model)
    main_menu(screen, mcts, game, board)





if __name__ == "__main__":
    main()