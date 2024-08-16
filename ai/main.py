from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import torch

from game.board import Board
from .neural_network import NeuralNetwork


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

n_actions = 7
n_observations = 200
agent = NeuralNetwork(n_observations, n_actions).to(device)

steps_done = 0

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


num_episodes = 3000

for i_episode in range(num_episodes):
    board = Board()

    total_score = 0
    for t in count():
        next_states = board.get_next_states()
        if len(next_states) == 0:
            print(f"{i_episode}: {t} ({total_score})")
            episode_durations.append(total_score)
            plot_durations()
            break
        best_state = agent.best_state([*next_states.values()])

        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        observation, points, reward, terminated = board.step(best_action)
        # print(terminated)
        total_score += points
        reward = torch.tensor([reward], device=device)

        next_state = None
        if not terminated:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

        state = torch.tensor(best_state, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        action = torch.tensor([action], device=device).unsqueeze(0)
        reward = torch.tensor([reward], device=device)
        agent.add_memory(state, action, next_state, reward)

        agent.train(device)

        board.display_current_state()

        if terminated:
            print(f"{i_episode}: {t} ({total_score})")
            episode_durations.append(total_score)
            plot_durations()
            break
