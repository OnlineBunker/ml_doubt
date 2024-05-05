import pygame
import random
import numpy as np
import tensorflow as tf

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BAR_WIDTH, BAR_HEIGHT = 10, 100
BALL_SIZE = 10
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BALL_SPEED = 5
BAR_SPEED = 5
REWARD_WIN = 1
REWARD_LOSS = -1
REWARD_NONE = 0
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1
MEMORY_SIZE = 2000
BATCH_SIZE = 32

# Initialize Pygame
pygame.init()

# Create the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pong")

# Define the player bar
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BAR_WIDTH, BAR_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = 30
        self.rect.y = SCREEN_HEIGHT // 2 - BAR_HEIGHT // 2

    def update(self, keys):
        if keys[pygame.K_UP]:
            self.rect.y -= BAR_SPEED
        if keys[pygame.K_DOWN]:
            self.rect.y += BAR_SPEED
        self.rect.y = max(0, min(SCREEN_HEIGHT - BAR_HEIGHT, self.rect.y))

# Define the AI bar using Deep Q-Network
class AI(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.model = self.build_model()
        self.memory = []
        self.last_state = None
        self.last_action = None
        self.image = pygame.Surface((BAR_WIDTH, BAR_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH - 30 - BAR_WIDTH
        self.rect.y = SCREEN_HEIGHT // 2 - BAR_HEIGHT // 2  

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(3)  # 3 actions: move up, stay still, move down
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='mse')
        return model

    def update(self, state, reward):
        if self.last_state is not None:
            self.memory.append((self.last_state, self.last_action, reward, state))
            if len(self.memory) > MEMORY_SIZE:
                del self.memory[0]
            if len(self.memory) >= BATCH_SIZE:
                self.train_model()
        self.last_state = state

    def train_model(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([s[0] for s in batch])
        actions = np.array([s[1] for s in batch])
        rewards = np.array([s[2] for s in batch])
        next_states = np.array([s[3] for s in batch])
        next_qs = self.model.predict(next_states)
        targets = rewards + DISCOUNT_FACTOR * np.max(next_qs, axis=1)
        targets_full = self.model.predict(states)
        targets_full[np.arange(len(actions)), actions] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)

    def select_action(self, state):
        if np.random.rand() < EPSILON:
            return random.randint(0, 2)
        else:
            return np.argmax(self.model.predict(np.array([state]))[0])

# Define the ball
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_SIZE, BALL_SIZE))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 2 - BALL_SIZE // 2
        self.rect.y = SCREEN_HEIGHT // 2 - BALL_SIZE // 2
        self.vx = BALL_SPEED * random.choice([-1, 1])
        self.vy = BALL_SPEED * random.choice([-1, 1])

    def update(self):
        self.rect.x += self.vx
        self.rect.y += self.vy

        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.vy *= -1

        if self.rect.colliderect(player.rect):
            self.vx *= -1
            self.rect.x = player.rect.right
        elif self.rect.colliderect(ai.rect):
            self.vx *= -1
            self.rect.x = ai.rect.left - BALL_SIZE
        elif self.rect.left <= 0:
            self.rect.x = SCREEN_WIDTH // 2 - BALL_SIZE // 2
            self.rect.y = SCREEN_HEIGHT // 2 - BALL_SIZE // 2
        elif self.rect.right >= SCREEN_WIDTH:
            self.rect.x = SCREEN_WIDTH // 2 - BALL_SIZE // 2
            self.rect.y = SCREEN_HEIGHT // 2 - BALL_SIZE // 2

# Create sprite groups
all_sprites = pygame.sprite.Group()
player = Player()
ai = AI()
ball = Ball()
all_sprites.add(player, ai, ball)

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()  
    player.update(keys)  

    state = [ball.rect.y, ai.rect.y]

    action = ai.select_action(state)

    ai.rect.y += (action - 1) * BAR_SPEED
    ai.rect.y = max(0, min(SCREEN_HEIGHT - BAR_HEIGHT, ai.rect.y))

    ball.update()

    if ball.rect.colliderect(player.rect):
        reward = REWARD_NONE
    elif ball.rect.colliderect(ai.rect):
        reward = REWARD_NONE
    elif ball.rect.left <= 0:
        reward = REWARD_LOSS
    elif ball.rect.right >= SCREEN_WIDTH:
        reward = REWARD_WIN
    else:
        reward = REWARD_NONE

    ai.update(state, reward)

    screen.fill(BLACK)
    all_sprites.draw(screen)
    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()
