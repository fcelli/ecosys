import tensorflow as tf
import keras


class ActorCritic(tf.keras.Model):
    '''Combined actor-critic network.'''
    def __init__(
            self,
            num_actions: int,
            num_hidden_units: int):
        '''Initialize model.'''
        super().__init__()
        
        self.common = keras.layers.Dense(num_hidden_units, activation='relu')
        self.actor = keras.layers.Dense(num_actions, activation='softmax')
        self.critic = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
