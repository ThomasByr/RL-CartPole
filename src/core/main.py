import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
from termcolor import colored

from tensorflow.python.keras import layers
from typing import List, Tuple

from ..cfg import Config

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

# Huber loss function
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

# Optimizer for the actor-critic loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Model save path & configuration

cfg = Config.cv1  #+ config
model_path = f"models/{cfg.to_str()}-model.ckpt"

# Constants
min_episodes_criterion = 1_000
max_episodes = 10_000
max_steps_per_episode = 1_000_000

reward_threshold = cfg.get_reward_threshold()
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Create the environment
env = cfg.to_gym()
env.observation_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")  # common network
        self.actor = layers.Dense(num_actions)  # actor network
        self.critic = layers.Dense(1)  # critic network

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


# Wrap OpenAI Gym"s `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episode(initial_state: tf.Tensor, model: tf.keras.Model,
                max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma*discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))

    return returns


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


@tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
               gamma: float, max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:

        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)

        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model"s parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


#%% training loop


def load_weights(model: tf.keras.Model) -> bool:
    loaded = False

    # try to load the model
    print(colored(f"Trying to load {cfg.to_str()} model weights", "blue"))
    try:
        model.load_weights(model_path)
        loaded = True
        print(colored("Loaded model from disk", "green"))
    except:
        print(colored("Could not load model from disk", "red"))

    return loaded


def train(model: tf.keras.Model):
    loaded = load_weights(model)

    # Keep last episodes reward : `deque` is a double-ended queue
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    if not loaded:
        with tqdm.trange(max_episodes) as t:
            for i in t:  # for each episode
                initial_state = tf.constant(env.reset(), dtype=tf.float32)  # get initial state
                episode_reward = int(train_step(initial_state, model, optimizer, gamma,
                                                max_steps_per_episode))  # run episode

                episodes_reward.append(episode_reward)  # add episode reward to the deque
                running_reward = statistics.mean(episodes_reward)  # calculate running reward

                t.set_description(f"Episode {i}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward > reward_threshold and i >= min_episodes_criterion:
                    break

        print(colored(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!", "green"))

        model.save_weights(model_path)  # save the model
        print(colored("Saved model to disk", "green"))


#%% render

from PIL import Image


def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
    screen = env.render(mode="rgb_array")
    im = Image.fromarray(screen)

    images = [im]

    state = tf.constant(env.reset(), dtype=tf.float32)
    for _ in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)  # add batch dimension
        action_probs, _ = model(state)  # get action probabilities
        action = np.argmax(np.squeeze(action_probs))  # get action

        state, _, done, _ = env.step(action)  # take action
        state = tf.constant(state, dtype=tf.float32)  # convert to tensor

        # render screen every step
        screen = env.render(mode="rgb_array")  # get screen
        images.append(Image.fromarray(screen))  # convert to PIL image

        if done:
            break

    return images


def export(model: tf.keras.Model):
    # Save GIF image
    images = render_episode(env, model, max_steps_per_episode)
    image_file = f"out/{cfg.to_str()}.gif"
    # loop=0: loop forever, duration=1: play each frame for 1ms
    images[0].save(fp=image_file, save_all=True, append_images=images[1:], loop=0, duration=1)

    print(colored(f"Saved GIF", "green"))


#%% entry point


def main():
    """
    Main entry point for the program.
    """
    # Set seed for experiment reproducibility
    seed = 42
    env.reset(seed=seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_actions = env.action_space.n  # 2 actions: left or right

    num_hidden_units = 1 << 7  # 2^7 = 128 hidden units

    model = ActorCritic(num_actions, num_hidden_units)

    train(model)
    export(model)
