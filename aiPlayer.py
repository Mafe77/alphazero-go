import torch
import numpy as np

from dlgo.network.GoNetwork import GoNetwork
from dlgo import goboard

class AIPlayer:
    def __init__(self, model_path, encoder, device='cuda'):
        self.encoder = encoder
        self.device = torch.device(device)

        self.model = GoNetwork(input_channels=encoder.num_planes, num_classes=361)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"AI model loaded from {model_path}")
        print(f"Using device: {self.device}")

    def select_move(self, game_state):
        # Encode board
        board_tensor = self.encoder.encode(game_state)

        # Convert to pytorch tensor and add batch dimension
        x = torch.from_numpy(board_tensor).float().unsqueeze(0)
        x = x.to(self.device)

        # Get model predictions
        with torch.no_grad():
            predictions = self.model(x)
            predictions = predictions.squeeze(0)

        # Convert to numpy
        move_probs = torch.softmax(predictions, dim=0).cpu().numpy()

        # Rank moves by probability
        ranked_moves = np.argsort(move_probs)[::-1]

        # Try moves in order of prob until theres a valid one
        for move_idx in ranked_moves:
            point = self.encoder.decode_point_index(move_idx)
            move = goboard.Move.play(point)
            if game_state.is_valid_move(move):
                return move
        
        # No valid moves found
        return goboard.Moves.pass_turn()

    