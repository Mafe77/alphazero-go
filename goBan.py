from collections import deque
from board import Board
from MCTS import MCTS
import numpy as np

class GoGame():
    def __init__(self, n=9):
        self.n = n
    
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
            + 1 to account for passing
        """
        return self.n * self.n + 1
    
    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # if the player is passing
        if action == self.n * self.n:
            return (board, -player)
        # init new board object
        b = Board(self.n)
        # copy current board
        b.pieces = np.copy(board)
        # convert action index into board coordinate
        move = (int(action/self.n), action%self.n)
        # execute the move
        b.execute_move(move, player)
        return (b.pieces, -player)
    
    def getValidMoves(self, board, player=1):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        valids = np.zeros(self.n * self.n + 1, dtype=np.uint8)
        legalMoves = b.get_legal_moves(player)

        if len(legalMoves) == 0:
            valids[-1] = 1  # only "pass" is valid
            return valids

        for x, y in legalMoves:
            valids[self.n * x + y] = 1

        return valids
    
    def getScore(self, board):
        n = board.shape[0]
        visited = np.zeros((n, n), dtype=bool)
        black_score = np.sum(board == -1)
        white_score = np.sum(board == 1)

        # Directions for neighbors
        directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for i in range(n):
            for j in range(n):
                if board[i,j] != 0 or visited[i,j]:
                    continue

                # BFS to find connected empty area
                queue = deque()
                queue.append((i,j))
                visited[i,j] = True
                territory = [(i,j)]
                bordering_colors = set()

                while queue:
                    x,y = queue.popleft()
                    for dx,dy in directions:
                        nx,ny = x+dx, y+dy
                        if 0 <= nx < n and 0 <= ny < n:
                            if board[nx,ny] == 0 and not visited[nx,ny]:
                                visited[nx,ny] = True
                                queue.append((nx,ny))
                                territory.append((nx,ny))
                            elif board[nx,ny] != 0:
                                bordering_colors.add(board[nx,ny])

                # If all bordering stones are same color, territory belongs to that color
                if len(bordering_colors) == 1:
                    color = bordering_colors.pop()
                    if color == 1:
                        white_score += len(territory)
                    else:
                        black_score += len(territory)

        return black_score, white_score

    def getGameEnded(self, board):
        """
        Determine if the game has ended.

        Returns:
            0      : game ongoing
            1      : black wins
            -1     : white wins
            1e-4   : draw
        """
        b = Board(self.n)
        b.pieces = np.copy(board)

        # If there are legal moves, game is still ongoing
        if len(b.get_legal_moves(1)) > 0 or len(b.get_legal_moves(-1)) > 0:
            return 0

        # Game ended, calculate score
        black_score, white_score = b.get_score()

        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return -1
        else:
            return 1e-4  # draw

    def getValueAndTerminated(self, board):
        if self.getGameEnded(board) == 1 or self.getGameEnded(board) == -1:
            return 1, True
        return 0, False
        

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board (numpy array)
            player: current player (1 or -1)

        Returns:
            canonicalBoard: board from current player's perspective
        """
        # Multiply the board by the player
        # If player = 1 (white), board stays the same
        # If player = -1 (black), all stones are inverted
        canonicalBoard = board * player
        return canonicalBoard


    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()
    
    def getOpponent(self, player):
        return -player
    
    def getOpponentValue(self, value):
        return -value

    @staticmethod
    def display(board):
        n = board.shape[0]

        for y in range(n):
            print(y, "|", end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1:
                    print("b ", end="")
                elif piece == 1:
                    print("W ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("   -----------------------")

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class GreedyGobangPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            black_score, white_score = self.game.getScore(nextBoard)
            candidates += [(-white_score, a)]
        candidates.sort()
        return candidates[0][1]

game = GoGame(9)  # Your Go game
board = game.getInitBoard()  # Get empty board

ai = RandomPlayer(game)      # Initialize random AI
player = 1            # You can be black (-1)
ai_color = -1                 # AI is white (1)

args = {
    'C': 1.41,
    'num_searches': 1000
}

mcts = MCTS(game, args)

current_player = 1 

while True:
    if current_player == ai_color:
        # action = ai.play(board)
        neutral_state = game.getCanonicalForm(board, current_player)
        mcts_probs = mcts.search(neutral_state, current_player)
        print(mcts_probs)
        action = np.argmax(mcts_probs)
    else:
        # Ask human for input (row and col)
        x = int(input("Row: "))
        y = int(input("Col: "))
        action = x * game.n + y

    black_score, white_score = game.getScore(board)
    print("Black:", black_score)
    print("White:", white_score)

    
    # Get valid moves and check if the action is allowed
    valids = game.getValidMoves(board, current_player)
    if valids[action] != 1:
        print("Invalid move! Try again.")
        continue
    
    # Update the board
    board, current_player = game.getNextState(board, current_player, action)
    
    # Print the board
    game.display(board)
    
    # Optional: check end of game
    if np.sum(game.getValidMoves(board, current_player)[:-1]) == 0:
        print("Game over!")
        break

