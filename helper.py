from consts import MARGIN, BOARD_SIZE, WINDOW_SIZE, CELL_SIZE

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