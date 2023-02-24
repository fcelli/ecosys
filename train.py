import tensorflow as tf
import keras
import collections
import statistics
import tqdm
from ecosys.environment import Ecosystem
from ecosys.models import ActorCritic
from ecosys.training import Trainer


# Environment
GRID_DIM = 10
N_RESOURCES = 20
# Model
N_HIDDEN = 64
LEARNING_RATE = 0.01
# Trainer
MIN_EPISODES = 1000
MAX_EPISODES = 30000
MAX_STEPS = 500
GAMMA = 0.99
REWARD_THRESHOLD = 270


def main():
    # Create environment
    env = Ecosystem(grid_dim=GRID_DIM, n_resources=N_RESOURCES)
    # Initialize ML model
    model = ActorCritic(
        num_actions=env.action_space.n,
        num_hidden_units=N_HIDDEN
    )
    # Initialize the Trainer
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    trainer = Trainer(env, model, optimizer)
    # Episode loop
    episodes_reward: collections.deque = collections.deque(maxlen=MAX_EPISODES)
    running_rewards: collections.deque = collections.deque(maxlen=MAX_EPISODES)
    t = tqdm.trange(MAX_EPISODES)
    for i in t:
        initial_state = env.reset()
        initial_state = tf.constant(initial_state, dtype=tf.int8)
        episode_reward = float(trainer.train_step(initial_state, GAMMA, MAX_STEPS))
        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)
        running_rewards.append(running_reward)
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)
        if running_reward > REWARD_THRESHOLD and i >= MIN_EPISODES:
            break
    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
    # Compile and save model
    model.compile()
    model.save('./data/models/ActorCritic.model')


if __name__ == '__main__':
    main()
