"""
Environment

User based environment to perform the CartPole task.
"""
#pylint: disable=[C0103, C0114, C0301, C0411, W0401, W0614, W0703, R0902, R0912, R0913, R0914, E1101]

import collections
import statistics
import pygame
from typing import List, Tuple

import gym
import numpy as np
import tensorflow as tf
import tqdm
import sys
from PIL import Image

from ..cfg import *
from .ac import ActorCritic

__all__ = ["Env"]

config: dict[str, Config] = {
    "cv0": Config.cv0,
    "0": Config.cv0,
    "cv1": Config.cv1,
    "1": Config.cv1,
}

WIDTH, HEIGHT = 600, 400  # screen size based on CartPole


class Env:
    """
    User Environment to perform the CartPole task.
    """

    # Small epsilon value for stabilizing division operations
    eps = np.finfo(np.float32).eps.item()

    # Huber loss function
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    # Optimizer for the actor-critic loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def __init__(self, *args: Config | Device | str | int, **kwargs: Config | Device | str | int) -> None:
        """
        Initialize the environment.

        ## Parameters
        ```py
        cfg: Config
            Configuration object.
        ```
        """

        params: dict[str, Config | Device] = {
            "cfg": Config.cv0,
            "device": Device.cpu,
        }

        # get params from args and kwargs
        for value in args:

            if isinstance(value, Config):
                params["cfg"] = value
            elif isinstance(value, Device):
                params["device"] = value
            elif isinstance(value, str):
                try:
                    params["cfg"] = config[value.lower()]
                except KeyError:
                    error(f"Unknown config: {value} ; defaulting to cv0")
            elif isinstance(value, int):
                try:
                    params["cfg"] = config[str(value)]
                except KeyError:
                    error(f"Unknown config: {value} ; defaulting to 0")

        for key, value in kwargs.items():

            if key not in params:
                continue

            if isinstance(value, Config):
                params[key] = value
            elif isinstance(value, Device):
                params[key] = value
            elif isinstance(value, str):
                try:
                    params[key] = config[value.lower()]
                except KeyError:
                    error(f"Unknown config: {value} ; defaulting to cv0")
            elif isinstance(value, int):
                try:
                    params[key] = config[str(value)]
                except KeyError:
                    error(f"Unknown config: {value} ; defaulting to 0")

        # assign params to class attributes
        self._cfg: Config = params["cfg"]
        self._device: Device = params["device"]

        self._model_path = f"models/{self.cfg.to_str()}-model.ckpt"

        # Constants
        self.min_episodes_criterion = 1_000
        self.max_episodes = 10_000  # should be large enough to ensure convergence
        self.max_steps_per_episode = 500  # max of 100 for v0 and 500 for v1

        self.reward_threshold = self.cfg.get_reward_threshold()
        self.running_reward = 0

        # Discount factor for future rewards
        self.gamma = 0.99

        # Create the environment
        self.env = self.cfg.to_gym()
        self.env.observation_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)

    def __str__(self) -> str:
        """Return a string representation of the environment."""
        return f"Env({self._cfg.to_str()})"

    @property
    def cfg(self) -> Config:  # pylint: disable=C0116
        return self._cfg

    @cfg.setter
    def cfg(self, _: Config) -> None:  # pylint: disable=C0116
        warn("Setting the environment configuration on runtime is not supported.")

    @property
    def device(self) -> Device:  # pylint: disable=C0116
        return self._device

    @device.setter
    def device(self, _: Device) -> None:  # pylint: disable=C0116
        warn("Setting the environment device on runtime is not supported.")

    @property
    def model_path(self) -> str:  # pylint: disable=C0116
        return self._model_path

    @model_path.setter
    def model_path(self, _: str) -> None:  # pylint: disable=C0116
        warn("Setting the model path manually is not supported.")

    # Wrap OpenAI Gym"s `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        """Returns state, reward and done flag given an action."""
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

    def run_episode(self, initial_state: tf.Tensor, model: tf.keras.Model,
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
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(self, rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
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
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))

        return returns

    def compute_loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self, initial_state: tf.Tensor, model: tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer, gamma: float,
                   max_steps_per_episode: int) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(initial_state, model, max_steps_per_episode)

            # Calculate expected returns
            returns = self.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model"s parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward

    def load_weights(self, model: tf.keras.Model) -> bool:
        """Try to load weights from the model path."""
        loaded = False

        # try to load the model
        debug(f"Trying to load {self.cfg.to_str()} model weights")
        try:
            model.load_weights(self.model_path)
            loaded = True
            info("Loaded model from disk")
        except Exception as e:
            if isinstance(e, tf.errors.NotFoundError):
                error("Could not find model weights on disk")
            else:
                fatal(f"Could not load model from disk: {e}")

        return loaded

    def train(self, model: tf.keras.Model):
        """Trains the model."""
        loaded = self.load_weights(model)

        # Keep last episodes reward : `deque` is a double-ended queue
        episodes_reward: collections.deque = collections.deque(maxlen=self.min_episodes_criterion)

        if not loaded:
            with tqdm.trange(self.max_episodes, file=sys.stdout) as t:
                i: int = -1
                for i in t:  # for each episode
                    initial_state = tf.constant(self.env.reset(), dtype=tf.float32)  # get initial state
                    episode_reward = int(
                        self.train_step(initial_state, model, self.optimizer, self.gamma,
                                        self.max_steps_per_episode))  # run episode

                    episodes_reward.append(episode_reward)  # add episode reward to the deque
                    running_reward = statistics.mean(episodes_reward)  # calculate running reward

                    t.set_description(f"Episode {i}")
                    t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                    if running_reward > self.reward_threshold and i >= self.min_episodes_criterion:
                        break

                debug(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!")

            model.save_weights(self.model_path)  # save the model
            info("Saved model to disk")

    def render_episode(self, env: gym.Env, model: tf.keras.Model, max_steps: int) -> List[Image.Image]:
        """Renders an episode of the environment."""
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

    def render_gif(self, model: tf.keras.Model) -> None:
        """Produces a GIF of the final episode."""
        # Save GIF image
        images = self.render_episode(self.env, model, self.max_steps_per_episode)
        image_file = f"out/{self.cfg.to_str()}.gif"
        # loop=0: loop forever, duration=x: play each frame for x ms
        images[0].save(fp=image_file, save_all=True, append_images=images[1:], loop=0, duration=0.2)

        info("Saved GIF")

    def interactive_run(self, model: tf.keras.Model) -> None:
        """
        Runs the model interactively.

        ## Parameters
        ```python
        model: tf.keras.Model
            The model to run.
        ```
        """
        pygame.init()

        debug("Running interactive mode...")

        all_actions = list(range(self.env.action_space.n))  # 0 for left, 1 for right
        is_running, should_tilt, tilt, released, warned = True, False, False, True, False
        tilt_fc, tilt_fc_max = 0, 6
        direction = ({0} | {1}).pop()  # do not make assumptions about the initial direction
        window = pygame.display.set_mode((WIDTH, HEIGHT))
        font = pygame.font.SysFont("Arial", 15)
        pygame.display.set_caption(self.cfg.get_name())

        state = tf.constant(self.env.reset(), dtype=tf.float32)
        clock = pygame.time.Clock()

        lr_l_text, lr_s_text, lr_r_text = "left", "|", "right"
        lr_s_surface = font.render(lr_s_text, True, (0, 0, 0))
        lr_l_surface = font.render(lr_l_text, True, (0, 0, 0))

        w0 = lr_l_surface.get_width() + 2
        w1 = lr_s_surface.get_width() + 2

        while is_running:
            clock.tick(60)  # limit to 60 FPS
            fps = clock.get_fps()

            # Handle events
            for event in pygame.event.get():

                # exit event
                if event.type == pygame.QUIT:
                    is_running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        is_running = False

                # mouse events
                elif event.type == pygame.MOUSEBUTTONDOWN and tilt_fc == 0 and released:
                    should_tilt = True
                    direction = 0 if pygame.mouse.get_pos()[0] < WIDTH / 2 else 1
                    released = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    released = True

            if should_tilt or tilt_fc > 0:
                tilt_fc += 1

            should_tilt = False

            tilt = False
            if 0 < tilt_fc <= tilt_fc_max:
                tilt = True
            elif tilt_fc > tilt_fc_max:
                tilt_fc = 0

            state = tf.expand_dims(state, 0)  # add batch dimension
            action_probs, _ = model(state)  # get action probabilities

            if tilt:
                action = all_actions[direction]  # get user action
            else:
                action = np.argmax(np.squeeze(action_probs))  # get action

            state, rating, done, _ = self.env.step(action)  # take action
            state = tf.constant(state, dtype=tf.float32)  # convert to tensor

            screen = self.env.render(mode="rgb_array")  # get screen
            img_surface = pygame.image.frombuffer(screen, (WIDTH, HEIGHT), "RGB")  # convert to pygame surface
            window.blit(img_surface, (0, 0))

            # Consistency warning
            if done and not warned and rating <= self.eps:
                warn("Episode failed and should end : consistency can no longer be guaranteed.")
                warned = True

            # Display frame rate and other info
            fps_text = f"fps: {fps:.0f}"
            info_text = "click the screen to tilt the cart"

            fps_text_surface = font.render(fps_text, True, (0, 0, 0))
            info_text_surface = font.render(info_text, True, (0, 0, 0))
            lr_r_surface = font.render(lr_r_text, True, (51, 255, 51) if tilt and direction == 1 else
                                       (255, 51, 51))
            lr_l_surface = font.render(lr_l_text, True, (51, 255, 51) if tilt and direction == 0 else
                                       (255, 51, 51))
            window.blit(fps_text_surface, (10, 10))
            window.blit(info_text_surface, (10, 30))

            window.blit(lr_r_surface, (WIDTH/2 + w1, 10))
            window.blit(lr_s_surface, (WIDTH / 2, 10))
            window.blit(lr_l_surface, (WIDTH/2 - w0, 10))

            pygame.display.flip()

        pygame.quit()

    def run(self) -> None:
        """Run the user environment."""
        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=cfg)
        tf.compat.v1.keras.backend.set_session(session)

        if (n := Device.nd()) == 0:
            debug("No GPU found.")
            if "gpu" in self.device.get_name():
                fatal("Can't run on GPU - please force CPU runmode or let it on auto-detect.")
        else:
            debug(f"Found {n} GPU(s).")

        info(f"Using {self.device.get_name()}.")

        with tf.device(self.device.get_name()):

            # Set seed for experiment reproducibility
            seed = 42
            self.env.reset(seed=seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)

            num_actions = self.env.action_space.n  # 2 actions: left or right

            num_hidden_units = 1 << 7  # 2^7 = 128 hidden units

            model = ActorCritic(num_actions, num_hidden_units)
            model.compile(optimizer=self.optimizer, loss=self.huber_loss)
            self.train(model)

            # self.render_gif(model)
            self.interactive_run(model)

        model.summary()
