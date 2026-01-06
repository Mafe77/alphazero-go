import pygame
import sys
import torch

from consts import *
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board
from dlgo.encoders.simple import SimpleEncoder
from aiPlayer import AIPlayer

class Game:
    def __init__(self, model_path):
        self.display_surface = pygame.display.get_surface()
        self.game = goboard.GameState.new_game(BOARD_SIZE)
        self.board = self.game.board
        self.last_move = None
        # self.draw_board()
        self.human_color = gotypes.Player.black
        self.ai_color = gotypes.Player.white
        self.encoder = SimpleEncoder((BOARD_SIZE, BOARD_SIZE))

        if model_path:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.ai_player = AIPlayer(model_path, self.encoder, device)
            self.ai_enabled = True

            print(f"Human plays as: {self.human_color}")
            print(f"AI plays as: {self.ai_color}")
        else:
            self.ai_player = None
            self.ai_enabled = False
            print("AI not enabled. Two player mode")
        
        self.hover_pos = None
        self.thinking = False
        
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


        blackPiece = pygame.image.load("assets/BlackPiece.png").convert_alpha()
        whitePiece = pygame.image.load("assets/WhitePiece.png")
        # Draw stones
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                point = gotypes.Point(row=row, col=col)
                stone = self.board.get(point)
                color = (0, 0, 0)

                if stone not in (gotypes.Player.black, gotypes.Player.white):
                    continue  # Empty space
                
                # Convert board coordinates → screen coordinates
                # Go: row 1 is bottom, Pygame: y increases downward
                x = BOARD_X + (col - 1) * CELL_SIZE
                y = BOARD_Y + (self.board.num_rows - row) * CELL_SIZE

                # Draw stone
                if stone == gotypes.Player.black:
                    self.display_surface.blit(blackPiece, (x - 20,y - 20))
                    color = (255, 255, 255)
                else:  # white stone
                    self.display_surface.blit(whitePiece, (x - 20,y - 20))
                    color = (0, 0, 0)
                
                if self.last_move == goboard.Move(point):
                    pygame.draw.circle(self.display_surface, color, (x, y), 6, width=3)
        
        # if self.hover_pos:
        #     x = BOARD_X + (self.hover_pos.col - 1) * CELL_SIZE
        #     y = BOARD_Y + (self.board.num_rows - self.hover_pos.row) * CELL_SIZE
            
        #     # Create semi-transparent surface
        #     hover_piece = blackPiece.copy()
        #     hover_piece.set_alpha(120)

        #     self.display_surface.blit(
        #         hover_piece,
        #         (x - STONE_RADIUS, y - STONE_RADIUS)
        #     )


    
    # def draw_ui(self):

    def make_ai_move(self):
        if not self.ai_enabled or self.thinking:
            return
        
        if self.game.next_player == self.ai_color:
            self.thinking = True
            self.draw_board()
            pygame.display.flip()

            move = self.ai_player.select_move(self.game)
            self.last_move = move
            # print("last ai:", self.last_move)

            self.game = self.game.apply_move(move)
            self.board = self.game.board
            
            self.thinking = False
        

        

    def run(self):
        clock = pygame.time.Clock()

        bg = pygame.image.load("assets/boardBG.png")
        pygame.mouse.set_visible(False)
        cursor_img = pygame.image.load("assets/cursor.png").convert_alpha()
        cursor_img_rect = cursor_img.get_rect()

        if self.ai_enabled and self.game.next_player == self.ai_color:
            self.make_ai_move()
        
        while True:
            cursor_img_rect.center = pygame.mouse.get_pos()            

            # adding board bg
            self.display_surface.blit(bg, (0, 0))
            self.draw_board()
            self.display_surface.blit(cursor_img, cursor_img_rect) 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and not self.thinking:
                    if self.game.next_player == self.human_color or not self.ai_enabled:                        
                        pos = self.get_board_position(event.pos)
                        if pos:
                            move = goboard.Move.play(pos)
                            if self.game.is_valid_move(move):
                                self.game = self.game.apply_move(move)
                                self.board = self.game.board
                                self.last_move = move

                                # pygame.time.wait(500)
                                self.make_ai_move()
                    
                
                elif event.type == pygame.MOUSEMOTION:
                    self.hover_pos = self.get_board_position(event.pos)
                    if self.hover_pos and self.board.get(self.hover_pos) is not None:
                        self.hover_pos = None
                        
            pygame.display.flip()
        