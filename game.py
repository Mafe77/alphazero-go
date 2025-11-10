import pygame
import numpy as np
from board import Board
from goBan import GoGame
from MCTS import MCTS

# Configuration
BOARD_SIZE = 9           # change this to 9 or 13 if you want smaller boards
WINDOW_SIZE = 800
MARGIN = 40               # margin around the grid
LINE_COLOR = (0, 0, 0)
BG_COLOR = (238, 178, 109)  # wooden-ish
STAR_POINT_COLOR = (0, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
HOVER_COLOR = (40, 120, 40, 150)

# Derived
CELL_COUNT = BOARD_SIZE - 1
CELL_SIZE = (WINDOW_SIZE - 2 * MARGIN) / CELL_COUNT
STONE_RADIUS = int(CELL_SIZE * 0.45)

pygame.init()

font = pygame.font.Font('freesansbold.ttf', 20)
medium_font = pygame.font.Font('freesansbold.ttf', 40)
big_font = pygame.font.Font('freesansbold.ttf', 50)

WIDTH = 1000
HEIGHT = 900

turn_step = 0
selection = 100
valid_moves = []

screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('AlphaGo Game')
clock = pygame.time.Clock()
running = True

game = GoGame(BOARD_SIZE)
board = game.getInitBoard()

# Helper functions
def to_screen(i, j):
    """Board coord (row i, col j) -> pixel (x, y)"""
    x = MARGIN + j * CELL_SIZE
    y = MARGIN + i * CELL_SIZE
    return int(round(x)), int(round(y))

def from_screen(x, y):
    """pixel -> nearest board coord (i, j) or None if outside reasonable area"""
    if x < MARGIN - CELL_SIZE/2 or x > WINDOW_SIZE - MARGIN + CELL_SIZE/2:
        return None
    if y < MARGIN - CELL_SIZE/2 or y > WINDOW_SIZE - MARGIN + CELL_SIZE/2:
        return None
    j = int(round((x - MARGIN) / CELL_SIZE))
    i = int(round((y - MARGIN) / CELL_SIZE))
    if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
        return i, j
    return None

def draw_board(surface):
    surface.fill(BG_COLOR)
    # grid lines
    for k in range(BOARD_SIZE):
        start = to_screen(0, k)
        end = to_screen(BOARD_SIZE-1, k)
        pygame.draw.line(surface, LINE_COLOR, start, end, 1)
        start = to_screen(k, 0)
        end = to_screen(k, BOARD_SIZE-1)
        pygame.draw.line(surface, LINE_COLOR, start, end, 1)
    # star points (hoshi) for standard sizes
    star_points = []
    if BOARD_SIZE == 19:
        pts = [3, 9, 15]
        star_points = [(r, c) for r in pts for c in pts]
    elif BOARD_SIZE == 13:
        pts = [3, 6, 9]
        star_points = [(r, c) for r in pts for c in pts]
    elif BOARD_SIZE == 9:
        pts = [2, 4, 6]
        star_points = [(r, c) for r in pts for c in pts]
    for (i,j) in star_points:
        x,y = to_screen(i,j)
        pygame.draw.circle(surface, STAR_POINT_COLOR, (x,y), max(3, int(CELL_SIZE*0.08)))

    # stones
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            val = board[i][j]
            if val != 0:
                x,y = to_screen(i,j)
                color = BLACK if val == 1 else WHITE
                pygame.draw.circle(surface, color, (x,y), STONE_RADIUS)
                # simple rim for white stones
                if val == 2:
                    pygame.draw.circle(surface, (0,0,0), (x,y), STONE_RADIUS, 1)

def draw_ui(surface, hover_pos):
    # draw a panel at bottom
    pygame.draw.rect(surface, (200,200,200), (0, WINDOW_SIZE, WINDOW_SIZE, 80))
    # current player
    text = "Black to play" if current_player == 1 else "White to play"
    t = big_font.render(text, True, BLACK)
    surface.blit(t, (10, WINDOW_SIZE + 8))
    # message
    if message:
        m = font.render(message, True, BLACK)
        surface.blit(m, (10, WINDOW_SIZE + 40))

    # Buttons: Pass, Reset
    pass_rect = pygame.Rect(WINDOW_SIZE - 200, WINDOW_SIZE + 10, 80, 40)
    reset_rect = pygame.Rect(WINDOW_SIZE - 100, WINDOW_SIZE + 10, 80, 40)
    pygame.draw.rect(surface, (180,180,180), pass_rect)
    pygame.draw.rect(surface, (180,180,180), reset_rect)
    surface.blit(font.render("Pass", True, BLACK), (pass_rect.x + 18, pass_rect.y + 10))
    surface.blit(font.render("Reset", True, BLACK), (reset_rect.x + 18, reset_rect.y + 10))
    return pass_rect, reset_rect

def draw_hover(surface, hover, current_player):
    if hover is None:
        return
    i, j = hover
    x, y = to_screen(i, j)
    color = (40,40,40,120) if current_player == 1 else (240, 240, 240, 120)
    s = pygame.Surface((STONE_RADIUS*2+4, STONE_RADIUS*2+4), pygame.SRCALPHA)
    pygame.draw.circle(s, (color), (STONE_RADIUS+2, STONE_RADIUS+2), STONE_RADIUS)
    surface.blit(s, (x - STONE_RADIUS -2, y - STONE_RADIUS -2))

hover = None
message = ""

args = {
    'C': 1.41,
    'num_searches': 10
}

mcts = MCTS(game, args)

current_player = 1
passes = 0
game_over = False

while running:
    clock.tick(60)
    # mouse position
    mx, my = pygame.mouse.get_pos()
    # for hover effect
    hover = from_screen(mx, my)
    valids = game.getValidMoves(board, current_player)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN and current_player == 1:            
            mx, my = event.pos
            if my >= WINDOW_SIZE:
                # UI clicked
                pass_rect, reset_rect = draw_ui(screen, hover)
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
    draw_board(screen)

    if hover:
        hi, hj = hover
        if board[hi][hj] == 0:
            draw_hover(screen, hover, current_player)

    # draw_board(screen)
    pass_rect, reset_rect = draw_ui(screen, hover)
    

    pygame.display.flip()

pygame.quit()