import math
import numpy as np

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state  # 2D board
        self.player = player  # current player to move at this node
        self.parent = parent
        self.action_taken = action_taken
        
        # binary vector of valid moves (1 = valid, 0 = invalid)
        self.expandable_moves = self.game.getValidMoves(self.state, self.player)
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0

    def select(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        epsilon = 1e-8
        q_value = 0.5 if child.visit_count == 0 else (child.value_sum / child.visit_count + 1) / 2
        ucb = q_value + self.args['C'] * math.sqrt(math.log(self.visit_count + 1) / (child.visit_count + epsilon))
        return ucb

    def expand(self):
        # pick a random valid move to expand
        move_index = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[move_index] = 0  # mark as used
        
        # get next board and next player
        next_state, next_player = self.game.getNextState(self.state, self.player, move_index)
        next_state = self.game.getCanonicalForm(next_state, next_player)
        
        child = Node(self.game, self.args, next_state, next_player, parent=self, action_taken=move_index)
        self.children.append(child)
        return child

    def simulate(self):
        rollout_state = self.state.copy()
        rollout_player = self.player

        while True:
            value, is_terminal = self.game.getValueAndTerminated(rollout_state)
            if is_terminal:
                # value is from current player's perspective
                if rollout_player != self.player:
                    value = self.game.getOpponentValue(value)
                return value

            valid_moves = self.game.getValidMoves(rollout_state, rollout_player)
            if np.sum(valid_moves) == 0:
                # no legal moves, pass
                action = self.game.getActionSize() - 1
            else:
                action = np.random.choice(np.where(valid_moves == 1)[0])
            
            rollout_state, rollout_player = self.game.getNextState(rollout_state, rollout_player, action)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        # invert value for opponent
        value = self.game.getOpponentValue(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state, player):
        root = Node(self.game, self.args, state, player)

        for _ in range(self.args['num_searches']):
            node = root

            # Selection
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.select()

            # Simulation / Expansion
            value, is_terminal = self.game.getValueAndTerminated(node.state)
            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            # Backpropagation
            node.backpropagate(value)

        # Compute action probabilities
        action_probs = np.zeros(self.game.getActionSize(), dtype=np.float32)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)

        return action_probs
