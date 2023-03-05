import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, point, block_size

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None
        self.trainer = None

    def get_state(self, game):
        head = game.snake[0]
        point_l = point(head.x - block_size, head.y)
        point_r = point(head.x + block_size, head.y)
        point_u = point(head.x, head.y - block_size)
        point_d = point(head.x, head.y + block_size)

        dir_l = game.direction == 'left'
        dir_r = game.direction == 'right'
        dir_u = game.direction == 'up'
        dir_d = game.direction == 'down'

        state = [
            # 1. Danger parameters
            # Straight
            (dir_r and game.collision(point_r)) or
            (dir_l and game.collision(point_l)) or
            (dir_u and game.collision(point_u)) or
            (dir_d and game.collision(point_d)),
            # Right
            (dir_u and game.collision(point_r)) or
            (dir_d and game.collision(point_l)) or
            (dir_l and game.collision(point_u)) or
            (dir_r and game.collision(point_d)),
            # Left
            (dir_d and game.collision(point_r)) or
            (dir_u and game.collision(point_l)) or
            (dir_r and game.collision(point_u)) or
            (dir_l and game.collision(point_d)),

            # 2. Move direction
            dir_l, dir_r, dir_u, dir_d,

            # 3. Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_ltm(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        self.trainer.train_step(zip(*batch))

    def train_stm(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random move generator
        self.epsilon = 100 - self.n_games
        next_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state_tensor)
            move = torch.argmax(prediction).item()
        next_move[move] = 1
        return next_move

def train():
    game = SnakeGame()
    agent = Agent()
    plot_scores = []
    mean_scores = []
    total_score = 0
    best_score = 0
    while True:
        prev_state = agent.get_state(game)
        next_move = agent.get_action(prev_state)
        reward, game_over, score = game.play_step(next_move)
        curr_state = agent.get_state(game)
        agent.train_stm(prev_state, next_move, reward, curr_state, game_over)
        agent.remember(prev_state, next_move, reward, curr_state, game_over)
        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_ltm()

            if score > best_score:
                best_score = score
                # agent.model.save()
            
            print(f"Try {agent.n_games} Score {score} Best {best_score}")

if __name__ == '__main__':
    train()