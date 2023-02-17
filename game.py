import pygame
pygame.init()
import numpy
from ecosys.environment import Ecosystem
import tensorflow as tf
import keras


FPS = 5
GRID_DIM = 10
N_RESOURCES = 20

def main():
    # Create environment and reset its state
    eco = Ecosystem(
        grid_dim=GRID_DIM,
        n_resources=N_RESOURCES
    )
    state = eco.reset()
    # Load ML model
    model = keras.models.load_model('./data/ActorCritic.model')
    # Display game screen
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ecosys")
    # Game loop
    clock = pygame.time.Clock()
    quit_button = False
    done = False
    while not (quit_button or done):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_button = True
        # Render environment to screen
        eco.render(screen)
        pygame.display.update()
        # Determine action
        state = tf.constant(state, dtype=tf.int8)
        state = tf.reshape(state, [tf.size(state)])
        state = tf.expand_dims(state, 0)
        action_probs, _ = model(state)
        action = numpy.argmax(numpy.squeeze(action_probs))
        state, reward, done = eco.step(action)
    # Quit game
    pygame.quit()

if __name__ == '__main__':
    main()
