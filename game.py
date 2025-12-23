import pygame
import sys

from consts import *
from dlgo import goboard


class Game:
    def __init__(self):
        self.display_surface = pygame.display.get_surface()
        self.game = goboard.GameState.new_game(BOARD_SIZE)
        self.board = self.game.board
        # self.draw_board()
    
    def draw_board(self):
        self.display_surface.fill(BOARD_COLOR)

        for row in range(self.board.num_rows):
            start_pos = (MARGIN, MARGIN + row * CELL_SIZE)
            end_pos = (MARGIN + (self.board.num_cols - 1) * CELL_SIZE, MARGIN + row * CELL_SIZE)
            pygame.draw.line(self.display_surface, LINE_COLOR, start_pos, end_pos, 2)
        
        for col in range(self.board.num_cols):
            start_pos = (MARGIN + col * CELL_SIZE, MARGIN)
            end_pos = (MARGIN + col * CELL_SIZE, MARGIN + (self.board.num_rows - 1) * CELL_SIZE)
            pygame.draw.line(self.display_surface, LINE_COLOR, start_pos, end_pos, 2)

    def run(self):
         while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # if event.type == pygame.MOUSEBUTTONDOWN:
                    # pos = get_board_position(event.pos, self.board)
                    # print(pos)
                    # move = goboard.Move.play(pos)
                    # if pos and self.board.is_valid_move(pos):
                    #     self.game = self.game.apply_move(move)                    
                
                # elif event.type == pygame.MOUSEMOTION:
                    # hover_pos = get_board_position(event.pos, game.board)
                    # if hover_pos and self.board.get(hover_pos) is not None:
                    #     hover_pos = None
            
            self.draw_board()
            pygame.display.flip()
        