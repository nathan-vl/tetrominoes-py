import random
import torch
import torch.nn as nn
import torch.optim as optim

from ai.memory import Memory
from ai.utils import Transition
from game.action import Action


class NeuralNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, discount=0.99, lr=1e-4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_observations, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, n_actions),
        )

        self.discount = discount
        self.memory = Memory(10000)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

        # Hiperparâmetros
        self.EPSILON = 1
        self.EPSILON_MIN = 0
        self.EPSILON_STOP_EP = 5000
        self.epsilon_decay = (self.EPSILON - self.EPSILON_MIN) / self.EPSILON_STOP_EP

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def add_memory(self, *args):
        self.memory.push(*args)

    def best_state(self, states):
        if random.random() <= self.EPSILON:
            return random.choice(states)

        max_score_state = None
        best_state = None
        for state in states:
            score = self(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).max(1).indices.view(1, 1)
            if max_score_state is None or score > max_score_state:
                max_score_state = score
                best_state = state
        return best_state

    def select_action(self, device, state):
        if random.random() <= self.EPSILON:
            return torch.tensor([[Action.sample()]], device=device, dtype=torch.long)
        with torch.no_grad():
            return self(state).max(1).indices.view(1, 1)

    def train(self, device, batch_size=50):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=device)

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
