import numpy as np
import tensorflow as tf
from ecosys.environment import EcosysEnv


class ActorCriticTrainer:
    def __init__(
        self,
        env: EcosysEnv,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer
    ):
        '''Initialize Trainer.'''
        # Environment
        self.env = env
        # ML model
        self.model = model
        # Optimizer
        self.optimizer = optimizer

    def run_episode(
        self,
        initial_state: tf.Tensor,
        max_steps: int
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        '''Run a single episode to collect training data'''
        # Initialize tensors containing the action probabilities, the critic values and the rewards
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # Start the episode loop
        initial_state_shape = initial_state.shape
        state = initial_state
        for t in tf.range(max_steps):
            # Flatten environment state
            state = tf.reshape(state, [tf.size(state)])
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)
            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)
            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            # Calculate action probabilities
            action_probs_t = tf.nn.softmax(action_logits_t)
            # Store critic values
            values = values.write(t, tf.squeeze(value))
            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])
            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)
            # Store reward
            rewards = rewards.write(t, reward)
            # Break loop if done is true
            if tf.cast(done, tf.bool):
                break
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        return action_probs, values, rewards

    def get_expected_return(
        self,
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True
    ) -> tf.Tensor:
        '''Compute expected returns per timestep.'''
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start from the end of `rewards` and accumulate reward sums into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        if standardize:
            # Small epsilon value for stabilizing division operations
            eps = np.finfo(np.float32).eps.item()
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
        return returns

    def compute_loss(
        self,
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor
    ) -> tf.Tensor:
        '''Computes the combined Actor-Critic loss.'''
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        advantage = returns - values
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = huber_loss(values, returns)
        return actor_loss + critic_loss

    @tf.function
    def train_step(
        self,
        initial_state: tf.Tensor,
        gamma: float,
        max_steps_per_episode: int
    ) -> tf.Tensor:
        '''Runs a model training step.'''
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(initial_state, max_steps_per_episode)
            # Calculate the expected returns
            returns = self.get_expected_return(rewards, gamma)
            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
            # Calculate the loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)
            # Compute the gradients from the loss
            grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        episode_reward = tf.math.reduce_sum(rewards)
        return episode_reward

    # Wrap Gym's `env.step` call as an operation in a TensorFlow function.
    # This allows it to be included in a callable TensorFlow graph.
    def env_step(
        self,
        action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Returns state, reward and terminated flag given an action.'''
        state, reward, terminated, _, _ = self.env.step(action)
        return (state.astype(np.int8), np.array(reward, np.float32), np.array(terminated, np.int8))

    def tf_env_step(
        self,
        action: tf.Tensor
    ) -> list[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.int8, tf.float32, tf.int8])
