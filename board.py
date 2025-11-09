import numpy as np
from collections import deque

'''
Board data:
white = 1, black = -1, empty = 0
first dimension is row, second dimension is col
    piece[2][1] is the square in row 3, col 2
Squares are stored and manipulated as (x,y) tuples
x is col, y is row

n = 3
piece[2][1] = 1
pieces = [
    [0,0,0],
    [0,0,0],
    [0,1,0]
]
'''

class Board:
    def __init__(self, n):
        self.n = n
        # init board using list comprehension
        self.pieces = [[0 for _ in range(n)] for _ in range(n)]

    def execute_move(self, move, player):
        x, y = move
        assert self.pieces[x, y] == 0, "Invalid move: spot occupied"
        self.pieces[x, y] = player
        self.remove_captured_stones(player, move)

    # ---------- Capture logic ----------
    def get_group_and_liberties(self, x, y):
        color = self.pieces[x, y]
        assert color != 0, "Starting point must be a stone"

        visited = set()
        liberties = set()
        group = set()
        queue = deque()
        queue.append((x, y))
        visited.add((x, y))
        group.add((x, y))

        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.n and 0 <= ny < self.n:
                    if self.pieces[nx, ny] == 0:
                        liberties.add((nx, ny))
                    elif self.pieces[nx, ny] == color and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        group.add((nx, ny))
        return group, liberties

    def remove_captured_stones(self, player, move):
        opponent = -player
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        for dx, dy in directions:
            nx, ny = move[0]+dx, move[1]+dy
            if 0 <= nx < self.n and 0 <= ny < self.n and self.pieces[nx, ny] == opponent:
                group, liberties = self.get_group_and_liberties(nx, ny)
                if len(liberties) == 0:
                    for gx, gy in group:
                        self.pieces[gx, gy] = 0

    # ---------- Legal moves ----------
    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x][y] == 0:
                    moves.add((x, y))
        return list(moves)

    def has_legal_moves(self):
        """Returns True if has legal move else False
        """
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self.pieces[x][y] == 0:
                    return True
        return False

    # ---------- Score ----------
    def get_score(self):
        visited = np.zeros((self.n, self.n), dtype=bool)
        black_score = np.sum(self.pieces == -1)
        white_score = np.sum(self.pieces == 1)
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for i in range(self.n):
            for j in range(self.n):
                if self.pieces[i, j] != 0 or visited[i, j]:
                    continue
                # BFS for empty region
                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                territory = [(i,j)]
                bordering_colors = set()

                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < self.n and 0 <= ny < self.n:
                            if self.pieces[nx, ny] == 0 and not visited[nx, ny]:
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                                territory.append((nx, ny))
                            elif self.pieces[nx, ny] != 0:
                                bordering_colors.add(self.pieces[nx, ny])
                if len(bordering_colors) == 1:
                    color = bordering_colors.pop()
                    if color == 1:
                        white_score += len(territory)
                    else:
                        black_score += len(territory)
        return black_score, white_score