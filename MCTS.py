import numpy as np
import math
import torch
import torch.nn.functional as F


class Node:
    def __init__(self, game, args, state, parent=None, prior=0, action_taken=None):
        self.game = game
        self.args = args
        
        self.state = state
        self.parent = parent
        self.action_taken = action_taken      # which move led to this node
        self.prior = prior                    # P(s,a) from policy network
        
        self.children = {}                    # action → Node
        
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, child):
        """
        U(s,a) = c * P(s,a) * sqrt(sum N) / (1 + N(s,a))
        Q(s,a) = scaled value
        """

        q = child.value()

        # convert [-1,1] → [0,1] for Go scoring
        # higher is better
        q = (q + 1) / 2

        u = self.args['C'] * child.prior * (
            math.sqrt(self.visit_count + 1) / (1 + child.visit_count)
        )
        return q + u

    def select_child(self):
        """Pick child with highest UCB."""
        best_action = None
        best_child = None
        best_ucb = -9999999
        
        for action, child in self.children.items():
            ucb = self.ucb_score(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
                best_action = action

        return best_child

    # def expand(self, policy):
    #     """Expand node using network policy."""
    #     self.is_expanded = True

    #     for action, prob in enumerate(policy):
    #         if prob <= 0:   # illegal move or zero probability
    #             continue

    #         next_state = self.game.getNextState(self.state, player=1, action=action)
    #         print("EXPAND", next_state)
    #         next_state = self.game.getCanonicalForm(next_state, player=-1)

    #         self.children[action] = Node(
    #             self.game,
    #             self.args,
    #             next_state,
    #             parent=self,
    #             prior=prob,
    #             action_taken=action
    #         )

    def expand(self, policy):
        self.children = []  # ensure clean list
        for action, prob in enumerate(policy):
            if prob <= 0:
                continue
            
            next_state = self.game.getNextState(self.state, 1, action)
            next_state = self.game.getCanonicalForm(next_state, -1)

            child = Node(
                game=self.game,
                args=self.args,
                state=next_state,
                parent=self,
                action_taken=action,
                prior=prob
            )        
                
            # self.children[action] = child
            self.children.append(child)

    def backpropagate(self, value):
        """Backup value and flip perspective."""
        self.visit_count += 1
        self.value_sum += value

        # flip player perspective
        value = self.game.getOpponentValue(value)

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
        # print("FUCKME4", root)

        # evaluate root first
        self.expand_node(root)

        for _ in range(self.args['num_searches']):
            node = root
            
            # 1. Selection
            while node.is_expanded and len(node.children) > 0:
                node = node.select_child()
            
            # 2. Evaluate node
            value, terminal = self.evaluate_node(node)

            # 3. Expand if not terminal
            if not terminal:
                self.expand_node(node)

            # 4. Backup
            node.backpropagate(value)

        # return visit count distribution
        full_probs = np.zeros(self.game.getActionSize(), dtype=np.float32)

        # print(root.children)
        # children *must* be: list[Node]
        for child in root.children:
            assert hasattr(child, "visit_count"), "ERROR: root.children must contain Node objects"
            full_probs[child.action_taken] = child.visit_count

        # Normalize safely
        total = full_probs.sum()

        if total == 0:
            # fallback: use legal moves
            legal = self.game.getValidMoves(root.state)
            full_probs = legal / legal.sum()
        else:
            full_probs /= total

        return full_probs


    @torch.no_grad()
    def evaluate_node(self, node):
        # print("FUCKME3", node.state)
        value, terminal = self.game.getValueAndTerminated(node.state)

        # flip because perspective is always "current player"
        value = self.game.getOpponentValue(value)

        return value, terminal

    @torch.no_grad()
    def expand_node(self, node):
        device = self.model.device  # <--- ALWAYS use model device
        # print("FUCKME5", node.state)

        # Encode state for neural network
        encoded = self.game.getEncodedState(node.state)
        encoded = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)

        policy, value = self.model(encoded)
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

        # mask policy with legal moves
        legal = self.game.getValidMoves(node.state)
        policy *= legal

        if policy.sum() == 0:
            policy = legal / legal.sum()

        node.expand(policy)


