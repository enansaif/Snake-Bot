import random
import pygame
pygame.init()
import numpy as np
from collections import namedtuple

font = pygame.font.SysFont('arial', 20)
colors = {'black' : (0, 0, 0), 'white' : (255,255,255),
          'red' : (255,0,0), 'green' : (0,255,0), 'blue' : (0,0,255)}
point = namedtuple('point', 'x, y')
block_size = 20

class SnakeGame:

    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.speed = 5

        # main game display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.direction = "right"
        self.head = point(self.w/2, self.h/2)
        self.snake = [self.head, point(self.head.x - block_size, self.head.y),
                      point(self.head.x - (2*block_size), self.head.y)]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_count = 0

    def place_food(self):
        #get position as multiple of block_size
        x = random.randint(0, (self.w-block_size) // block_size) * block_size
        y = random.randint(0, (self.h-block_size) // block_size) * block_size
        self.food = point(x, y)
        while self.food in self.snake:
            self.place_food()
    
    def move(self, action):
        x = self.head.x
        y = self.head.y

        # if action, bot will set the new direction
        # else it will take the player's direction
        if action:
            # [straigth, right, left] 100, 010, 001
            clock_wise = ["right", "down", "left", "up"]
            idx = clock_wise.index(self.direction)
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx] # continue on straight direction
            if np.array_equal(action, [0, 1, 0]):
                new_idx = (idx + 1) % 4 # clockwise next direction
                new_dir = clock_wise[new_idx]
            else:
                new_idx = (idx - 1) % 4 # clockwise prev direction
                new_dir = clock_wise[new_idx]
            self.direction = new_dir

        if self.direction == "right":
            x += block_size
        elif self.direction == "left":
            x -= block_size
        elif self.direction == "up":
            y -= block_size
        elif self.direction == "down":
            y += block_size

        self.head = point(x, y)

    def collision(self, p=None):
        # boundary collision
        p = self.head if not p else p
        if (p.x > self.w - block_size or p.x < 0 or 
            p.y > self.h - block_size or p.y < 0):
            return True
        # snake collision
        if p in self.snake[1:]:
            return True
        return False

    def update_ui(self):
        self.display.fill(colors['black'])
        for p in self.snake:
            pygame.draw.rect(self.display, colors['white'], 
                             pygame.Rect(p.x, p.y, block_size, 
                                         block_size))
        pygame.draw.rect(self.display, colors['green'], 
                         pygame.Rect(self.food.x, self.food.y, 
                                     block_size, block_size))
        
        text = font.render(f"Score:{self.score}", True, colors['white'])
        self.display.blit(text, (0, 0))
        pygame.display.flip()
    
    def play_step(self, action=None):
        # collect user input 
        self.frame_count += 1
        for even in pygame.event.get():
            if even.type == pygame.QUIT:
                pygame.quit()
                quit()
            if even.type == pygame.KEYDOWN:
                if even.key == pygame.K_LEFT:
                    self.direction = "left"
                elif even.key == pygame.K_RIGHT:
                    self.direction = "right"
                elif even.key == pygame.K_UP:
                    self.direction = "up"
                elif even.key == pygame.K_DOWN:
                    self.direction = "down"

        # move
        self.move(action)
        self.snake.insert(0, self.head)

        # check for game over
        reward = 0
        game_over = False
        # collision detection and bot inactivity detection
        if self.collision() or self.frame_count > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # place new food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()

        # update ui/clock
        self.update_ui()
        self.clock.tick(self.speed)

        return reward, game_over, self.score

if __name__ == '__main__':
    game = SnakeGame()
    while True:
        _, game_over, score = game.play_step()
        if game_over:
            break
    print(f"Score:{score}")
    pygame.quit()
