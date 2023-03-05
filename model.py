import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, filename="model.pth"):
        folder_path = "./model"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        filename = os.path.join(folder_path, filename)
        torch.save(self.state_dict(), filename)

class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train(self, prev_state, action, reward, next_state, game_over):
        prev_state = torch.tensor(np.array(prev_state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(prev_state.shape) == 1:
            prev_state = torch.unsqueeze(prev_state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        
        pred = self.model(prev_state)
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
