import matplotlib.pyplot as plt
import torch
from goBan import GoGame
from GoNet import GoNet
from consts import BOARD_SIZE

print(torch.__version__)
game = GoGame(BOARD_SIZE)

state = game.getInitBoard()
# state = game.get_next_state(state, 2, 1)
# state = game.get_next_state(state, 7, -1)

# print(state)

encoded_state = game.getEncodedState(state)
action_size = game.getActionSize()

print(encoded_state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

model = GoNet(BOARD_SIZE ,action_size, 6, 64)

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value, policy)

plt.bar(range(action_size), policy)
plt.show()