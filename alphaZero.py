import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm.notebook import trange
from MCTS import MCTS
from goBan import GoGame
from GoNet import GoNet


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.getInitBoard()   # state is a 2D numpy array

        while True:
            # canonical state from current player's perspective
            neutral_state = self.game.getCanonicalForm(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.getActionSize(), p=action_probs)

            # IMPORTANT: getNextState returns (next_board, next_player)
            next_board, next_player = self.game.getNextState(state, player, action)

            # update state and player correctly
            state = next_board
            player = next_player

            # value and terminal should be called with the board array only
            value, is_terminal = self.game.getValueAndTerminated(state)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    # hist_outcome = value from perspective of hist_player
                    # assume `value` returned above is from the perspective of the player who just moved / or as documented
                    # If value is from the perspective of current player at terminal, we need to convert appropriately:
                    # If hist_player == player (final player), hist_outcome = value else flip.
                    hist_outcome = value if hist_player == player else self.game.getOpponentValue(value)

                    returnMemory.append((
                        self.game.getEncodedState(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            # DO NOT flip player here — already updated above
            # player = self.game.getOpponent(player)  <-- remove this

                
    def train(self, memory):
        random.shuffle(memory)
        
        device = next(self.model.parameters()).device  # <<< FIX

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx+self.args['batch_size']] 
            state, policy_targets, value_targets = zip(*sample)

            state = torch.tensor(np.array(state), dtype=torch.float32, device=device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


game = GoGame(9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GoNet(9, game.getActionSize()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    'C': 2,
    'num_searches': 100,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64
}

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()