import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from ai.memory import Memory
from ai.neural_network import NeuralNetwork
from ai.utils import Transition
from game.action import Action

TAU = 0.5


class TetrominoesAgent:
    def __init__(self, device, discount=0.99, lr=1e-4, batch_size=50):
        self.neural_network = NeuralNetwork().to(device)

        self.memory = Memory(100000)

        self.device = device
        self.discount = discount
        self.optimizer = optim.AdamW(
            self.neural_network.parameters(),
            lr=lr,
            amsgrad=True,
        )

        self.loss_fn = nn.SmoothL1Loss()

        # Hiperparâmetros
        self.batch_size = batch_size
        self.EPSILON = 1
        self.EPSILON_MIN = 0
        self.EPSILON_STOP_EP = 250
        self.epsilon_decay = (self.EPSILON - self.EPSILON_MIN) / self.EPSILON_STOP_EP

    def add_memory(self, *args):
        self.memory.push(*args)

    def best_state(self, states):
        max_score_state = 0
        best_state = None
        best_action = None
        for state, action, reward in states:
            if best_state is None or reward >= max_score_state:
                max_score_state = reward
                best_state = state
                best_action = action
        return state, best_action, reward

    def select_action(self, state):
        if random.random() <= self.EPSILON:
            return torch.tensor(
                [[Action.sample()]],
                device=self.device,
            )
        with torch.no_grad():
            return self.neural_network(state).max(1).indices.view(1, 1)

    def recall(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminated_batch = torch.cat(batch.terminated)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminated_batch,
        )

    def estimate(self, state, action):
        current = self.neural_network(state)[np.arange(0, self.batch_size), action]
        return current

    @torch.no_grad
    def target(self, reward, next_state, terminated):
        next_state_q = self.neural_network(next_state)
        best_action = torch.argmax(next_state_q, 1)
        next = self.neural_network(next_state, target=True)[
            np.arange(0, self.batch_size),
            best_action,
        ]
        target = (reward + (1 - terminated.float()) * self.discount * next).float()
        return target

    def update_q(self, estimate, target):
        loss = self.loss_fn(estimate, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(
            self.neural_network.linear_relu_stack.parameters(), 100
        )
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        current_net_state_dict = self.neural_network.linear_relu_stack.state_dict()
        target_net_state_dict = self.neural_network.target.state_dict()

        for key in current_net_state_dict:
            target_net_state_dict[key] = current_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.neural_network.target.load_state_dict(target_net_state_dict)

        self.neural_network.target.load_state_dict(
            self.neural_network.linear_relu_stack.state_dict()
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        self.sync_target()

        state, action, reward, next_state, terminated = self.recall()

        estimate = self.estimate(state, action)
        target = self.target(reward, next_state, terminated)
        loss = self.update_q(estimate, target)

        return (estimate.mean().item(), loss)
