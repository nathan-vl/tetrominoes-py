import random
import torch
import torch.optim as optim
import torch.nn as nn

from ai.memory import Memory
from ai.neural_network import NeuralNetwork
from ai.utils import Transition
from game.action import Action


class TetrominoesAgent:
    def __init__(self, device, discount=0.99, lr=1e-4):
        self.neural_network = NeuralNetwork(200, 7)

        self.memory = Memory(10000)

        self.device = device
        self.discount = discount
        self.optimizer = optim.AdamW(self.neural_network.parameters(), lr=lr, amsgrad=True)

        # Hiperparâmetros
        self.EPSILON = 1
        self.EPSILON_MIN = 0
        self.EPSILON_STOP_EP = 50
        self.epsilon_decay = (self.EPSILON - self.EPSILON_MIN) / self.EPSILON_STOP_EP

    def add_memory(self, *args):
        self.memory.push(*args)

    def best_state(self, states):
        if random.random() <= self.EPSILON:
            return random.choice(states)

        max_score_state = None
        best_state = None
        for state in states:
            score = (
                self.neural_network(
                    torch.tensor(
                        state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                )
                .max(1)
                .indices.view(1, 1)
            )
            if max_score_state is None or score > max_score_state:
                max_score_state = score
                best_state = state
        return best_state

    def select_action(self, state):
        if random.random() <= self.EPSILON:
            return torch.tensor(
                [[Action.sample()]], device=self.device, dtype=torch.long
            )
        with torch.no_grad():
            return self.neural_network(state).max(1).indices.view(1, 1)

    def estimate(self, state, action):
        return self.neural_network(state, model="online")[torch.zeros(50), action]

    def train(self, batch_size=50):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.neural_network(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)

        # with torch.no_grad():
        #     next_state_values[non_final_mask] = (
        #         target_net(non_final_next_states).max(1).values
        #     )
        # expected_state_action_values = (
        #     next_state_values * self.discount
        # ) + reward_batch

        # criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        # self.optimizer.step()

        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON -= self.epsilon_decay
