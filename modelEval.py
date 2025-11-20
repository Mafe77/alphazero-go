import matplotlib.pyplot as plt
import torch
from goBan import GoGame
from GoNet import GoNet
from consts import BOARD_SIZE

print(torch.__version__)
game = GoGame(BOARD_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = game.getInitBoard()
state, _ = game.getNextState(state, 1, 40)
state, _ = game.getNextState(state, -1, 47)
state, _ = game.getNextState(state, 1, 51)


# print(state)

encoded_state = game.getEncodedState(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = GoNet(9, game.getActionSize())
model.load_state_dict(torch.load('model_2.pt', map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value)

print(state)
print(tensor_state)

plt.bar(range(game.getActionSize()), policy)
plt.show()