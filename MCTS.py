import math
import numpy as np
import torch

class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state          # canonical state
        self.player = player        # player to move in canonical form (- always +1 POV)
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

        # get valid moves for this state
        self.valid_moves = game.getValidMoves(self.state, self.player)

    def is_fully_expanded(self):
        return len(self.children) == np.sum(self.valid_moves)

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q = 0
        else:
            q = child.value_sum / child.visit_count

        u = self.args['C'] * child.prior * math.sqrt(self.visit_count + 1) / (1 + child.visit_count)

        return q + u
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if self.valid_moves[action] == 0:
                continue

            # apply move as player +1 (canonical player)
            next_state, next_player = self.game.getNextState(self.state, self.player, action)

            # ALWAYS convert to canonical perspective (+1 POV)
            next_state = self.game.getCanonicalForm(next_state, next_player)

            child = Node(
                self.game,
                self.args,
                next_state,
                next_player,
                parent=self,
                action_taken=action,
                prior=prob
            )

            self.children.append(child)

        return self.children[0]

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

        # always invert value when going to parent
        value = -value

        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, root_state, root_player=1):
        # start with canonical perspective
        state = self.game.getCanonicalForm(root_state, root_player)

        root = Node(self.game, self.args, state, player=1)

        for _ in range(self.args['num_searches']):
            node = root

            # Selection
            while node.children and node.is_fully_expanded():
                node = node.select()

            # Evaluate node
            value, terminal = self.game.getValueAndTerminated(node.state)

            if not terminal:
                # Neural network evaluation
                encoded = self.game.getEncodedState(node.state)
                encoded = torch.tensor(encoded).unsqueeze(0).float()

                policy, value_tensor = self.model(encoded)
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

                # mask invalid moves
                policy *= node.valid_moves
                policy_sum = np.sum(policy)
                if policy_sum > 0:
                    policy /= policy_sum
                else:
                    policy = node.valid_moves / np.sum(node.valid_moves)

                value = value_tensor.item()

                # Expand children
                node = node.expand(policy)

            # Backpropagation
            node.backpropagate(value)

        # Compute action probabilities
        action_probs = np.zeros(self.game.getActionSize())
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs


