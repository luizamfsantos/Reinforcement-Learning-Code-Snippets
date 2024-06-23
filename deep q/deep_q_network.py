import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Parameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 100000
episodes = 1000
seed = 42

# Replay Memory to store experiences
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Neural Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q Network Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(memory_size)
        self.model = (state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epsilon = epsilon  # Exploration rate

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        # Random generator
        rng = np.random.default_rng(seed)
        # Epsilon-greedy action selection
        if rng.random() < self.epsilon:
            return rng.choice(self.action_size)  # Explore: select a random action
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())  # Exploit: select the action with max Q-value

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor([done])

            # Compute Q-value for the current state
            q_values = self.model(state)
            q_value = q_values[0, action]

            # Compute the target Q-value
            next_q_values = self.target_model(next_state)
            next_q_value = reward + (1 - done) * gamma * torch.max(next_q_values)

            # Compute loss
            loss = nn.MSELoss()(q_value, next_q_value.detach())

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# Main function
if __name__ == "__main__":
    env = gym.make('CartPole-v1')  # Initialize the CartPole environment
    state_size = env.observation_space.shape[0]  # Size of the state space
    action_size = env.action_space.n  # Size of the action space
    agent = DQNAgent(state_size, action_size)  # Create DQN agent

    for e in range(episodes):
        state = env.reset()  # Reset environment to start a new episode
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)  # Select an action
            next_state, reward, done, _ = env.step(action)  # Take the action and observe the next state and reward
            reward = reward if not done else -10  # Penalize if episode ends
            next_state = np.reshape(next_state, [1, state_size])
            agent.memory.push(state, action, reward, next_state, done)  # Store the experience in replay memory
            state = next_state
            if done:
                agent.update_target_model()  # Update the target model
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            agent.train(batch_size)  # Train the agent
