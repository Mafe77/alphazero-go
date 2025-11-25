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
        self.pieces = np.zeros((n, n), dtype=np.int8)
        # print("pieces", self.pieces)

    def execute_move(self, move, player):
        x, y = move
        # print("EXECUTE")
        assert self.pieces[x][y] == 0, "Invalid move: spot occupied"
        self.pieces[x][y] = player
        self.remove_captured(x, y, player, self.pieces)

    def empty_board(self, n):
        self.pieces = np.zeros((n, n), dtype=np.int8)

    # ---------- Capture logic ----------
    def get_group_and_liberties(self, x, y, pieces=None):
        """Return the connected group and its liberties."""
        if pieces is None:
            pieces = self.pieces

        color = pieces[x][y]
        assert color != 0, "Starting point must be a stone"

        visited = set()
        group = set()
        liberties = set()
        queue = [(x, y)]
        visited.add((x, y))
        group.add((x, y))

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        while queue:
            cx, cy = queue.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.n and 0 <= ny < self.n:
                    if pieces[nx][ny] == 0:
                        liberties.add((nx, ny))
                    elif pieces[nx][ny] == color and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        group.add((nx, ny))

        return group, liberties

    def remove_captured(self, x, y, color, pieces=None):
        """Remove opponent stones with zero liberties."""
        if pieces is None:
            pieces = self.pieces

        opponent = -color
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n and pieces[nx][ny] == opponent:
                group, libs = self.get_group_and_liberties(nx, ny, pieces)
                if len(libs) == 0:
                    for gx, gy in group:
                        pieces[gx][gy] = 0

    # ---------------- Helper functions for legality ----------------
    def get_group_and_liberties_sim(self, pieces, x, y):
        """Same as get_group_and_liberties but explicitly for a given board array."""
        return self.get_group_and_liberties(x, y, pieces)

    def move_captures_opponent(self, pieces, x, y, color):
        """Check if placing a stone captures any opponent stones."""
        opponent = -color
        captured = False
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.n and 0 <= ny < self.n and pieces[nx][ny] == opponent:
                group, libs = self.get_group_and_liberties_sim(pieces, nx, ny)
                if len(libs) == 0:
                    captured = True
        return captured

    # ---------------- get legal moves ----------------
    def get_legal_moves(self, color):
        legal = []
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x][y] != 0:
                    continue

                temp = np.copy(self.pieces)
                temp[x][y] = color

                # Remove captured opponent stones
                self.remove_captured(x, y, color, pieces=temp)

                # Check suicide: if this group has no liberties and doesn't capture opponent, illegal
                group, libs = self.get_group_and_liberties_sim(temp, x, y)
                if len(libs) == 0 and not self.move_captures_opponent(self.pieces, x, y, color):
                    continue

                legal.append((x, y))
        return legal
