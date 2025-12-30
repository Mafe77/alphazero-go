import pygame
import sys

from consts import *
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board

class Game:
    def __init__(self):
        self.display_surface = pygame.display.get_surface()
        self.game = goboard.GameState.new_game(BOARD_SIZE)
        self.board = self.game.board
        # self.draw_board()
    
    def get_board_position(self, mouse_pos):
        x, y = mouse_pos

        # Calculate which intersection is closest
        col = round((x - BOARD_X) / CELL_SIZE) + 1
        row = self.board.num_rows - round((y - BOARD_Y) / CELL_SIZE)

        # Check if within board bounds
        if 1 <= row <= self.board.num_rows and 1 <= col <= self.board.num_cols:
            # Actual pixel position of the intersection
            actual_x = BOARD_X + (col - 1) * CELL_SIZE
            actual_y = BOARD_Y + (self.board.num_rows - row) * CELL_SIZE

            # Distance from mouse click to intersection
            distance = ((x - actual_x) ** 2 + (y - actual_y) ** 2) ** 0.5

            if distance <= 20:  # 20 pixel tolerance
                return gotypes.Point(row=row, col=col)

        return None


    def draw_board(self):
        # self.display_surface.fill(BOARD_COLOR)

        board_width = (self.board.num_cols - 1) * CELL_SIZE
        board_height = (self.board.num_rows - 1) * CELL_SIZE

        pygame.draw.rect(
            self.display_surface,
            BOARD_COLOR,
            (BOARD_X, BOARD_Y, board_width, board_height)
        )

        for row in range(self.board.num_rows):
            start_pos = (BOARD_X, BOARD_Y + row * CELL_SIZE)
            end_pos = (
                BOARD_X + (self.board.num_cols - 1) * CELL_SIZE,
                BOARD_Y + row * CELL_SIZE
            )
            pygame.draw.line(self.display_surface, LINE_COLOR, start_pos, end_pos, 2)
        
        for col in range(self.board.num_cols):
            start_pos = (BOARD_X + col * CELL_SIZE, BOARD_Y)
            end_pos = (
                BOARD_X + col * CELL_SIZE,
                BOARD_Y + (self.board.num_rows - 1) * CELL_SIZE
            )
            pygame.draw.line(self.display_surface, LINE_COLOR, start_pos, end_pos, 2)

        # Thick right border
        pygame.draw.line(
            self.display_surface,
            LINE_COLOR,
            (BOARD_X + board_width + 4, BOARD_Y + 2),
            (BOARD_X + board_width + 4, BOARD_Y + board_height),
            6  # thicker
        )

        pygame.draw.line(
            self.display_surface,
            LINE_COLOR,
            (BOARD_X + board_width + 8, BOARD_Y + 8),
            (BOARD_X + board_width + 8, BOARD_Y + board_height - 6),
            6  # thicker
        )

        # Thick bottom border
        pygame.draw.line(
            self.display_surface,
            LINE_COLOR,
            (BOARD_X + 4, BOARD_Y + board_height + 4),
            (BOARD_X + board_width, BOARD_Y + board_height + 4),
            6  # thicker
        )

        pygame.draw.line(
            self.display_surface,
            LINE_COLOR,
            (BOARD_X + 10, BOARD_Y + board_height + 8),
            (BOARD_X + board_width - 6, BOARD_Y + board_height + 8),
            6  # thicker
        )

        # Thick top border
        pygame.draw.line(
            self.display_surface,
            LINE_COLOR,
            (BOARD_X, BOARD_Y),
            (BOARD_X + board_width + 2, BOARD_X - 10),
            6  # thicker
        )

        # Thick left border
        pygame.draw.line(
            self.display_surface,
            LINE_COLOR,
            (BOARD_X, BOARD_Y),
            (BOARD_X, BOARD_Y + board_height + 1),
            6  # thicker
        )


        blackPiece = pygame.image.load("assets/BlackPiece.png")
        whitePiece = pygame.image.load("assets/WhitePiece.png")
        # Draw stones
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                stone = self.board.get(gotypes.Point(row=row, col=col))

                if stone not in (gotypes.Player.black, gotypes.Player.white):
                    continue  # Empty space

                # Convert board coordinates → screen coordinates
                # Go: row 1 is bottom, Pygame: y increases downward
                x = BOARD_X + (col - 1) * CELL_SIZE
                y = BOARD_Y + (self.board.num_rows - row) * CELL_SIZE

                # Draw stone
                if stone == gotypes.Player.black:
                    # pygame.draw.circle(
                    #     self.display_surface,
                    #     BLACK_STONE,
                    #     (x, y),
                    #     STONE_RADIUS
                    # )
                    self.display_surface.blit(blackPiece, (x - 20,y - 20))
                else:  # white stone
                    self.display_surface.blit(whitePiece, (x - 20,y - 20))


    
    # def draw_ui(self):
        

        

    def run(self):
        bg = pygame.image.load("assets/boardBG.png")
        while True:
            # adding board bg
            self.display_surface.blit(bg, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.get_board_position(event.pos)
                    # print(pos)
                    move = goboard.Move.play(pos)
                    if pos and self.board.is_valid_move(pos):
                        self.game = self.game.apply_move(move)
                        self.board = self.game.board            
                        # print_board(self.board)        
                
                elif event.type == pygame.MOUSEMOTION:
                    hover_pos = self.get_board_position(event.pos)
                    if hover_pos and self.board.get(hover_pos) is not None:
                        hover_pos = None
            
            self.draw_board()
            pygame.display.flip()
        