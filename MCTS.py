import numpy as np
import math
import torch
import torch.nn.functional as F


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0.0):
        self.game = game
        self.args = args
        
        self.state = state
        self.parent = parent
        self.action_taken = action_taken  # move that led to this node
        self.prior = float(prior)         # P(s,a)
        
        self.children = {}                
        self.visit_count = 0
        self.value_sum = 0.0

        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, child):
        """
        PUCT:
        Q + c * P * sqrt(N) / (1 + n)
        """

        Q = child.value()

        U = (
            self.args['C']
            * child.prior
            * math.sqrt(self.visit_count + 1)
            / (1 + child.visit_count)
        )

        return Q + U

    def select_child(self):
        """Select child with highest (Q+U)."""

        return max(self.children.items(), key=lambda item: self.ucb_score(item[1]))[1]

    def expand(self, policy):
        """Create children for legal moves."""
        self.is_expanded = True

        legal = self.game.getValidMoves(self.state)

        for action, prob in enumerate(policy):
            if legal[action] == 0 or prob <= 0:
                continue

            next_state, next_player = self.game.getNextState(self.state, 1, action)
            next_state = self.game.getCanonicalForm(next_state, next_player)

            self.children[action] = Node(
                game=self.game,
                args=self.args,
                state=next_state,
                parent=self,
                action_taken=action,
                prior=prob
            )

    def backpropagate(self, value):
        """Backup value and flip sign for opponent."""
        self.visit_count += 1
        self.value_sum += value

        value = -value    # flip for opponent at each level
        
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, root_state):
        root = Node(self.game, self.args, root_state)

        # expand root
        self.expand_node(root)

        for _ in range(self.args['num_searches']):
            node = root

            # 1. Selection
            while node.is_expanded and len(node.children) > 0:
                node = node.select_child()

            # 2. Evaluate state
            terminal_value, is_terminal = self.game.getValueAndTerminated(node.state)

            if not is_terminal:
                self.expand_node(node)

            # 3. Backup (canonical = player 1 POV, so use +value)
            node.backpropagate(terminal_value)

        # Convert visit counts to probabilities
        action_probs = np.zeros(self.game.getActionSize(), dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        if action_probs.sum() == 0:
            legal = self.game.getValidMoves(root.state)
            return legal / legal.sum()

        return action_probs / action_probs.sum()

    @torch.no_grad()
    def expand_node(self, node):
        device = self.model.device

        encoded = self.game.getEncodedState(node.state)
        encoded = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)

        policy_logits, value = self.model(encoded)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # mask illegal moves
        legal = self.game.getValidMoves(node.state)
        policy *= legal

        if policy.sum() == 0:
            policy = legal / legal.sum()
        else:
            policy /= policy.sum()

        node.expand(policy)