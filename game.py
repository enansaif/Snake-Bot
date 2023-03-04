from collections import namedtuple
import random
import pygame
pygame.init()

font = pygame.font.SysFont('arial', 20)
colors = {'black' : (0, 0, 0), 'white' : (255,255,255),
          'red' : (255,0,0), 'green' : (0,255,0), 'blue' : (0,0,255)}
point = namedtuple('point', 'x, y')
block_size = 20

class Game:

    def __init__(self, width=1280, height=720):
        self.w = width
        self.h = height
        self.score = 0
        self.speed = 8

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
    
    def move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == "right":
            x += block_size
        elif direction == "left":
            x -= block_size
        elif direction == "up":
            y -= block_size
        elif direction == "down":
            y += block_size
        self.head = point(x, y)

    def collision(self):
        # boundary collision
        if (self.head.x > self.w - block_size or self.head.x < 0 or 
            self.head.y > self.h - block_size or self.head.y < 0):
            return True
        # snake collision
        if self.head in self.snake[1:]:
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
    
    def play_step(self):
        # collect user input 
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
        self.move(self.direction)
        self.snake.insert(0, self.head)

        # check for game over
        game_over = False
        if self.collision():
            game_over = True
            return game_over, self.score
            
        # place new food
        if self.head == self.food:
            self.score += 1
            self.speed += 1
            self.place_food()
        else:
            self.snake.pop()

        # update ui/clock
        self.update_ui()
        self.clock.tick(self.speed)

        return game_over, self.score

if __name__ == '__main__':
    game = Game()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print(f"Score:{score}")
    pygame.quit()
