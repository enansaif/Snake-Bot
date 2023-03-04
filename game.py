import random
import pygame
from collections import namedtuple

colors = {'black' : (0, 0, 0),
          'white' : (255,255,255),
          'red' : (255,0,0),
          'green' : (0,255,0),
          'blue' : (0,0,255)}

point = namedtuple('point', 'x, y')
speed = 40
block_size = 20
pygame.init()
font = pygame.font.SysFont('arial', 20)
class Game:

    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.score = 0
        # main game display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = "right"
        self.head = point(self.w/2, self.h/2)
        self.snake = [self.head, point(self.head.x - block_size, self.head.y),
                      point(self.head.x - (2*block_size), self.head.y)]
        self.food = None
        self.place_food()
        

    def place_food(self):
        #get position as multiple of block_size
        x = random.randint(0, (self.w-block_size) // block_size) * block_size
        y = random.randint(0, (self.h-block_size) // block_size) * block_size
        self.food = point(x, y)
        while self.food in self.snake:
            self.place_food()

    def play_step(self):
        # collect user input 
        # move
        # check if game over
        # place new food
        # update ui/clock
        self.update_ui()
        self.clock.tick(speed)
        # return game over/score
        game_over = False
        return game_over, self.score
    
    def update_ui(self):
        self.display.fill(colors['black'])
        for p in self.snake:
            pygame.draw.rect(self.display, colors['white'], pygame.Rect(p.x, p.y, block_size, block_size))
        pygame.draw.rect(self.display, colors['green'], pygame.Rect(self.food.x, self.food.y, block_size, block_size))
        
        text = font.render(f"Score:{self.score}", True, colors['white'])
        self.display.blit(text, [0, 0])
        pygame.display.flip()

if __name__ == '__main__':
    game = Game()

    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print(f"Score:{score}")
    pygame.quit()
