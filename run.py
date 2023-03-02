import numpy
import keras
import gym
import tensorflow as tf


def main():
    # Create environment and reset its state
    env = gym.make('ecosys.env:Ecosys-v0', render_mode='human')
    state, _ = env.reset()
    env.render()
    # Load ML model
    model = keras.models.load_model('./data/models/ActorCritic.model')
    terminated = False
    while not terminated:
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
        state, _, terminated, _, _ = env.step(action)
        # Render environment
        env.render()
    # Close the environment
    env.close()


if __name__ == '__main__':
    main()
