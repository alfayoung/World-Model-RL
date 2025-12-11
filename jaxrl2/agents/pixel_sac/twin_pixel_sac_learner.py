"""Twin Pixel SAC Learner for Real-World Digital-Twin Double-Q Learning."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.critic_updater import update_critic
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter', 'aug_next', 'num_cameras', 'update_twin'))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic_real: TrainState,
    critic_twin: TrainState,
    target_critic_real_params: Params,
    target_critic_twin_params: Params,
    temp: TrainState,
    batch_real: TrainState,
    batch_twin: TrainState,
    discount: float,
    tau: float,
    target_entropy: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    update_twin: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, Params, Params, TrainState, Dict[str, float]]:
    """
    Update both real and twin critics along with the shared actor.

    Args:
        update_twin: Whether to update the twin critic in this iteration
    """
    info = {}

    # ========== Real Critic Update ==========
    # Data augmentation for real batch
    aug_pixels = batch_real['observations']['pixels']
    aug_next_pixels = batch_real['next_observations']['pixels']

    if batch_real['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch_real['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set(
                        (color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                    )
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations_real = batch_real['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch_real = batch_real.copy(add_or_replace={'observations': observations_real})

    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch_real['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set(
                        (color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                    )
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations_real = batch_real['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch_real = batch_real.copy(add_or_replace={'next_observations': next_observations_real})

    # Update real critic
    rng, key = jax.random.split(rng)
    target_critic_real = critic_real.replace(params=target_critic_real_params)
    new_critic_real, critic_real_info = update_critic(
        key, actor, critic_real, target_critic_real, temp, batch_real,
        discount, critic_reduction=critic_reduction
    )
    new_target_critic_real_params = soft_target_update(new_critic_real.params, target_critic_real_params, tau)

    # Add prefix to real critic info
    for k, v in critic_real_info.items():
        info[f'real_{k}'] = v

    # ========== Twin Critic Update (conditional) ==========
    new_critic_twin = critic_twin
    new_target_critic_twin_params = target_critic_twin_params

    if update_twin:
        # Data augmentation for twin batch
        aug_pixels_twin = batch_twin['observations']['pixels']
        aug_next_pixels_twin = batch_twin['next_observations']['pixels']

        if batch_twin['observations']['pixels'].squeeze().ndim != 2:
            rng, key = jax.random.split(rng)
            aug_pixels_twin = batched_random_crop(key, batch_twin['observations']['pixels'])

            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_pixels_twin = aug_pixels_twin.at[:,:,:,i*3:(i+1)*3].set(
                            (color_transform(key, aug_pixels_twin[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                        )
                else:
                    aug_pixels_twin = (color_transform(key, aug_pixels_twin.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

        observations_twin = batch_twin['observations'].copy(add_or_replace={'pixels': aug_pixels_twin})
        batch_twin = batch_twin.copy(add_or_replace={'observations': observations_twin})

        if aug_next:
            rng, key = jax.random.split(rng)
            aug_next_pixels_twin = batched_random_crop(key, batch_twin['next_observations']['pixels'])
            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_next_pixels_twin = aug_next_pixels_twin.at[:,:,:,i*3:(i+1)*3].set(
                            (color_transform(key, aug_next_pixels_twin[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                        )
                else:
                    aug_next_pixels_twin = (color_transform(key, aug_next_pixels_twin.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
            next_observations_twin = batch_twin['next_observations'].copy(
                add_or_replace={'pixels': aug_next_pixels_twin})
            batch_twin = batch_twin.copy(add_or_replace={'next_observations': next_observations_twin})

        # Update twin critic
        rng, key = jax.random.split(rng)
        target_critic_twin = critic_twin.replace(params=target_critic_twin_params)
        new_critic_twin, critic_twin_info = update_critic(
            key, actor, critic_twin, target_critic_twin, temp, batch_twin,
            discount, critic_reduction=critic_reduction
        )
        new_target_critic_twin_params = soft_target_update(new_critic_twin.params, target_critic_twin_params, tau)

        # Add prefix to twin critic info
        for k, v in critic_twin_info.items():
            info[f'twin_{k}'] = v

    # ========== Actor Update (uses real critic) ==========
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key, actor, new_critic_real, temp, batch_real,
        critic_reduction=critic_reduction
    )

    # ========== Temperature Update ==========
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    # Combine all info
    info.update(actor_info)
    info.update(alpha_info)

    return rng, new_actor, new_critic_real, new_critic_twin, new_target_critic_real_params, new_target_critic_twin_params, new_temp, info


class TwinPixelSACLearner(Agent):
    """
    Twin Pixel SAC Learner with separate critics for real and twin environments.

    This implements the Real-World Digital-Twin Double-Q Learning algorithm where:
    - Q_real is trained on transitions from env_real
    - Q_twin is trained on transitions from env_twin (digital twin)
    - Both critics share the same actor network
    """

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter=True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1):
        """
        Initialize Twin Pixel SAC with dual critics.
        """

        self.aug_next = aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_real_key, critic_twin_key, temp_key = jax.random.split(rng, 5)

        # ========== Encoder Definition ==========
        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                     softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                  softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                  softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        # ========== Actor Network ==========
        policy_def = LearnedStdTanhNormalPolicy(hidden_dims, self.action_dim,
                                               dropout_rate=dropout_rate,
                                               low=-action_magnitude, high=action_magnitude)

        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck)
        print("Actor architecture:")
        print(actor_def)
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)

        # ========== Real Critic Network ==========
        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck)
        print("Real Critic architecture:")
        print(critic_def)
        critic_real_def_init = critic_def.init(critic_real_key, observations, actions)

        critic_real_params = critic_real_def_init['params']
        critic_real_batch_stats = critic_real_def_init['batch_stats'] if 'batch_stats' in critic_real_def_init else None
        critic_real = TrainState.create(apply_fn=critic_def.apply,
                                       params=critic_real_params,
                                       tx=optax.adam(learning_rate=critic_lr),
                                       batch_stats=critic_real_batch_stats)
        target_critic_real_params = copy.deepcopy(critic_real_params)

        # ========== Twin Critic Network ==========
        # Use the same architecture but separate parameters
        print("Twin Critic architecture:")
        print(critic_def)
        critic_twin_def_init = critic_def.init(critic_twin_key, observations, actions)

        critic_twin_params = critic_twin_def_init['params']
        critic_twin_batch_stats = critic_twin_def_init['batch_stats'] if 'batch_stats' in critic_twin_def_init else None
        critic_twin = TrainState.create(apply_fn=critic_def.apply,
                                       params=critic_twin_params,
                                       tx=optax.adam(learning_rate=critic_lr),
                                       batch_stats=critic_twin_batch_stats)
        target_critic_twin_params = copy.deepcopy(critic_twin_params)

        # ========== Temperature ==========
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)

        self._rng = rng
        self._actor = actor
        self._critic_real = critic_real
        self._critic_twin = critic_twin
        self._target_critic_real_params = target_critic_real_params
        self._target_critic_twin_params = target_critic_twin_params
        self._temp = temp

        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)

        print(f'target_entropy: {self.target_entropy}')
        print(f'critic_reduction: {self.critic_reduction}')
        print('TwinPixelSACLearner initialized with dual critics (real + twin)')

    def update(self, batch_real: FrozenDict, batch_twin: Optional[FrozenDict] = None,
               update_twin: bool = True) -> Dict[str, float]:
        """
        Update the agent with batches from real and twin buffers.

        Args:
            batch_real: Batch from D_real (real environment transitions)
            batch_twin: Batch from D_twin (twin environment transitions)
            update_twin: Whether to update the twin critic in this iteration

        Returns:
            Dictionary of training metrics
        """
        # If no twin batch provided, create a dummy batch (won't be used if update_twin=False)
        if batch_twin is None:
            batch_twin = batch_real

        new_rng, new_actor, new_critic_real, new_critic_twin, new_target_critic_real, new_target_critic_twin, new_temp, info = _update_jit(
            self._rng,
            self._actor,
            self._critic_real,
            self._critic_twin,
            self._target_critic_real_params,
            self._target_critic_twin_params,
            self._temp,
            batch_real,
            batch_twin,
            self.discount,
            self.tau,
            self.target_entropy,
            self.critic_reduction,
            self.color_jitter,
            self.aug_next,
            self.num_cameras,
            update_twin,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic_real = new_critic_real
        self._critic_twin = new_critic_twin
        self._target_critic_real_params = new_target_critic_real
        self._target_critic_twin_params = new_target_critic_twin
        self._temp = new_temp

        return info

    def get_twin_q_value(self, observations: Dict, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Get Q-value from twin critic (used for trajectory scoring).

        Args:
            observations: Observation dictionary
            actions: Actions

        Returns:
            Q-values from twin critic (shape: [batch, num_qs])
        """
        input_collections = {'params': self._critic_twin.params}
        q_values = self._critic_twin.apply_fn(input_collections, observations, actions)
        if self.critic_reduction == 'min':
            return float(q_values.min())
        elif self.critic_reduction == 'mean':
            return float(q_values.mean())
        else:
            raise NotImplementedError()

    def get_real_q_value(self, observations: Dict, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Get Q-value from real critic (used for trajectory scoring).

        Args:
            observations: Observation dictionary
            actions: Actions

        Returns:
            Q-values from twin critic (shape: [batch, num_qs])
        """
        input_collections = {'params': self._critic_real.params}
        q_values = self._critic_real.apply_fn(input_collections, observations, actions)
        if self.critic_reduction == 'min':
            return float(q_values.min())
        elif self.critic_reduction == 'mean':
            return float(q_values.mean())
        else:
            raise NotImplementedError()

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        """Evaluation uses the real critic."""
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        """Visualization uses the real critic."""
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import make_visual, np_unstack

        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]

                # Use real critic for visualization
                input_collections = {'params': self._critic_real.params}
                q_value = self._critic_real.apply_fn(input_collections, obs_dict, action)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))

        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        """Save both critics and their target networks."""
        save_dict = {
            'critic_real': self._critic_real,
            'critic_twin': self._critic_twin,
            'target_critic_real_params': self._target_critic_real_params,
            'target_critic_twin_params': self._target_critic_twin_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        """Restore checkpoint including both critics."""
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic_real = output_dict['critic_real']
        self._critic_twin = output_dict['critic_twin']
        self._target_critic_real_params = output_dict['target_critic_real_params']
        self._target_critic_twin_params = output_dict['target_critic_twin_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)
