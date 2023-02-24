import os
import glob
import pygame
import numpy
import keras
from ecosys.environment import Ecosystem
import tensorflow as tf


FPS = 5
GRID_DIM = 10
N_RESOURCES = 20
SAVE_FRAMES = True


def main():
    # Initialize pygame
    pygame.init()
    # Create environment and reset its state
    env = Ecosystem(
        grid_dim=GRID_DIM,
        n_resources=N_RESOURCES
    )
    state = env.reset()
    # Load ML model
    model = keras.models.load_model('./data/models/ActorCritic.model')
    # Display screen
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ecosys")
    # Clear or create images folder
    if SAVE_FRAMES:
        if os.path.isdir('./data/images'):
            files = glob.glob('./data/images/screenshot_*.jpeg')
            for f in files:
                os.remove(f)
        else:
            os.mkdir('./data/images')
    # Start loop
    clock = pygame.time.Clock()
    quit_button = False
    done = False
    frame_count = 0
    while not (quit_button or done):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_button = True
        # Render environment to screen
        env.render(screen)
        pygame.display.update()
        # Convert state to Tensor
        state = tf.constant(state, dtype=tf.int8)
        # Flatten state Tensor
        state = tf.reshape(state, [tf.size(state)])
        # Add batch dimension for compatibility
        state = tf.expand_dims(state, 0)
        # Determine action probabilities
        action_probs, _ = model(state)
        # Take action with highest probability
        action = numpy.argmax(numpy.squeeze(action_probs))
        # Take next environment step
        state, reward, done = env.step(action)
        # Save frame images
        if SAVE_FRAMES:
            pygame.image.save(screen, f'./data/images/screenshot_{frame_count}.jpeg')
        frame_count += 1
    # Quit game
    pygame.quit()


if __name__ == '__main__':
    main()
