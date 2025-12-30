import pygame
import sys

from consts import *

from dlgo import gotypes

# Initialize Pygame
pygame.init()


# Calculate window size
WINDOW_WIDTH = CELL_SIZE * (BOARD_SIZE - 1) + 2 * MARGIN
WINDOW_HEIGHT = CELL_SIZE * (BOARD_SIZE - 1) + 2 * MARGIN

def get_board_position(mouse_pos, board):
    """
    Convert mouse position to board coordinates
    
    Args:
        mouse_pos: (x, y) tuple of mouse position
        board: your board object
    
    Returns:
        gotypes.Point or None if click is outside board
    """
    x, y = mouse_pos
    
    # Calculate which intersection is closest
    col = round((x - MARGIN) / CELL_SIZE) + 1
    row = board.num_rows - round((y - MARGIN) / CELL_SIZE)
    
    # Check if within board bounds
    if 1 <= row <= board.num_rows and 1 <= col <= board.num_cols:
        # Check if click is close enough to intersection (within 20 pixels)
        actual_x = MARGIN + (col - 1) * CELL_SIZE
        actual_y = MARGIN + (board.num_rows - row) * CELL_SIZE
        distance = ((x - actual_x) ** 2 + (y - actual_y) ** 2) ** 0.5
        
        if distance <= 20:  # 20 pixel tolerance
            return gotypes.Point(row=row, col=col)
    
    return None


def draw_board(screen, board, hover_pos=None):
    """
    Draw the Go board with pygame
    
    Args:
        screen: pygame display surface
        board: your board object with num_rows, num_cols, and get() method
        hover_pos: optional Point to show preview stone
    """
    # Fill background
    screen.fill(BOARD_COLOR)
    
    # Draw grid lines
    for row in range(board.num_rows):
        start_pos = (MARGIN, MARGIN + row * CELL_SIZE)
        end_pos = (MARGIN + (board.num_cols - 1) * CELL_SIZE, MARGIN + row * CELL_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, 2)
    
    for col in range(board.num_cols):
        start_pos = (MARGIN + col * CELL_SIZE, MARGIN)
        end_pos = (MARGIN + col * CELL_SIZE, MARGIN + (board.num_rows - 1) * CELL_SIZE)
        pygame.draw.line(screen, LINE_COLOR, start_pos, end_pos, 2)
    
    # Draw star points (for 19x19 board)
    if board.num_rows == 19 and board.num_cols == 19:
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for row, col in star_points:
            x = MARGIN + (col - 1) * CELL_SIZE
            y = MARGIN + (board.num_rows - row) * CELL_SIZE
            pygame.draw.circle(screen, LINE_COLOR, (x, y), 5)
    
    # Draw stones
    for row in range(1, board.num_rows + 1):
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            
            if stone != gotypes.Player.black and stone != gotypes.Player.white:
                continue  # Empty space
            
            # Calculate position (row 1 is at bottom in Go, but top in pygame)
            x = MARGIN + (col - 1) * CELL_SIZE
            y = MARGIN + (board.num_rows - row) * CELL_SIZE
            
            # Draw stone
            if stone == gotypes.Player.black:
                pygame.draw.circle(screen, BLACK_STONE, (x, y), STONE_RADIUS)
            elif stone == gotypes.Player.white:
                pygame.draw.circle(screen, WHITE_STONE, (x, y), STONE_RADIUS)
                pygame.draw.circle(screen, LINE_COLOR, (x, y), STONE_RADIUS, 2)  # Border
    
    # Draw coordinate labels
    font = pygame.font.Font(None, 24)
    
    # Column labels
    cols = "ABCDEFGHJKLMNOPQRST" 
    for col in range(board.num_cols):
        label = font.render(cols[col], True, TEXT_COLOR)
        x = MARGIN + col * CELL_SIZE - label.get_width() // 2
        y = MARGIN - 30
        screen.blit(label, (x, y))
        # Also at bottom
        screen.blit(label, (x, MARGIN + (board.num_rows - 1) * CELL_SIZE + 15))
    
    # Row labels
    for row in range(1, board.num_rows + 1):
        label = font.render(str(row), True, TEXT_COLOR)
        x = MARGIN - 30
        y = MARGIN + (board.num_rows - row) * CELL_SIZE - label.get_height() // 2
        screen.blit(label, (x, y))
        # Also on right side
        x_right = MARGIN + (board.num_cols - 1) * CELL_SIZE + 15
        screen.blit(label, (x_right, y))
    
    # Draw hover preview stone (semi-transparent)
    if hover_pos:
        x = MARGIN + (hover_pos.col - 1) * CELL_SIZE
        y = MARGIN + (board.num_rows - hover_pos.row) * CELL_SIZE
        
        # Create semi-transparent surface
        preview_surface = pygame.Surface((STONE_RADIUS * 2, STONE_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(preview_surface, (*BLACK_STONE , 128), (STONE_RADIUS, STONE_RADIUS), STONE_RADIUS)
        screen.blit(preview_surface, (x - STONE_RADIUS, y - STONE_RADIUS))

