from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import torch

from ai.agent import TetrominoesAgent
from game.board import Board


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

agent = TetrominoesAgent(device)

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

episode_scores = []
episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(scores_t.numpy(), color='green')
    plt.plot(durations_t.numpy(), )

    MEDIA_LENGTH = 50
    if len(scores_t) >= MEDIA_LENGTH:
        means = scores_t.unfold(0, MEDIA_LENGTH, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(MEDIA_LENGTH - 1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
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
    print(f"Episode: {i_episode}")
    no_score_delta = 0
    no_score_delta_limit = 500

    for t in count():
        state = board.current_state()
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        action = agent.select_action(state)

        observation, points, reward, terminated = board.step(action)

        # action = torch.tensor([action], dtype=torch.int32, device=device).unsqueeze(0)
        observation = torch.tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        reward = torch.tensor([reward], device=device).unsqueeze(0)
        terminated = torch.tensor([terminated], device=device).unsqueeze(0)
        agent.add_memory(state, action, observation, reward, terminated)
        # print(action[0][0].item(), end='')

        """
        As vezes a IA aprende como utilizar o sistema de giro de forma a não
        ganhar pontos mas continuar o jogo de forma indefinida. Esse trecho
        limita a quantidade de vezes que faz isso sem ganhar pontos para não
        interromper o processo.
        """
        if points < 10:
            no_score_delta += 1
        else:
            no_score_delta = 0

        total_score += points

        if terminated:
            episode_scores.append(board.get_game_score())
            episode_durations.append(t + 1)
            plot_durations()
            break
        if no_score_delta >= no_score_delta_limit:
            episode_scores.append(board.get_game_score())
            episode_durations.append(t + 1)
            plot_durations()
            break
        
    board.display_current_state()
    # print()
    agent.train()

    if agent.EPSILON > agent.EPSILON_MIN:
        agent.EPSILON -= agent.epsilon_decay
