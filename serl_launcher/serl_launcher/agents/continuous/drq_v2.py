import copy
from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop


class DrQv2Agent(SACAgent):
    """
    DrQ-v2 Agent: https://arxiv.org/abs/2107.09645
    Uses DDPG (Deterministic Policy) with Data Augmentation.
    """
    
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.99,
        soft_target_update_rate: float = 0.005,  # tau
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers (No temperature optimizer)
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                image_keys=image_keys,
            ),
        )

    @classmethod
    def create_drq_v2(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False, 
            "std_parameterization": "fixed", 
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        """
        Create a new pixel-based DrQ-v2 agent.
        """
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder
            encoders = {
                image_key: SmallEncoder(features=(32, 64, 128, 256), kernel_sizes=(3, 3, 3, 3), strides=(2, 2, 2, 2), padding="VALID", pool_method="avg", bottleneck_dim=256, spatial_block_size=8, name=f"encoder_{image_key}")
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs
            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](pooling_method="spatial_learned_embeddings", num_spatial_blocks=8, bottleneck_dim=256, name=f"encoder_{image_key}")
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder, resnetv1_configs
            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](pre_pooling=True, name="pretrained_encoder")
            encoders = {
                image_key: PreTrainedResNetEncoder(pooling_method="spatial_learned_embeddings", num_spatial_blocks=8, bottleneck_dim=256, pretrained_encoder=pretrained_encoder, name=f"encoder_{image_key}")
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        # For DrQ-v2 (Deterministic), we use fixed_std=0.0
        policy_kwargs["tanh_squash_distribution"] = True 
        policy_kwargs["std_parameterization"] = "fixed"
        # Use a per-action std vector (all zeros for deterministic DrQ-v2) to satisfy
        # MultivariateNormalDiag shape requirements.
        policy_kwargs["fixed_std"] = jnp.zeros((actions.shape[-1],), dtype=jnp.float32)

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )
        
        if encoder_type == "resnet-pretrained":
             from serl_launcher.utils.train_utils import load_resnet10_params
             agent = load_resnet10_params(agent, image_keys)

        return agent

    def data_augmentation_fn(self, rng, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        
        # 1. Data Augmentation for Target Q Calculation
        # DrQ-v2: Averaging over M=2 augmentations for target Q
        M = 2 
        rng, *aug_rngs = jax.random.split(rng, M + 1)
        
        all_target_qs = []
        
        for i in range(M):
            aug_next_obs = self.data_augmentation_fn(aug_rngs[i], batch["next_observations"])
            
            # Target Policy (Deterministic)
            target_action_dist = self.state.apply_fn(
                {"params": self.state.target_params},
                aug_next_obs,
                name="actor",
                rngs={"dropout": rng},
                train=False,
            )
            next_actions = target_action_dist.mode()

            # Target Critic
            target_next_qs_local = self.forward_target_critic(
                aug_next_obs,
                next_actions,
                rng=rng,
            )
            
            # Min over ensemble
            target_next_min_q_local = target_next_qs_local.min(axis=0)
            all_target_qs.append(target_next_min_q_local)
            
        # Average the target Qs
        avg_target_next_q = jnp.mean(jnp.stack(all_target_qs, axis=0), axis=0)
        
        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * avg_target_next_q
        )
        
        # 2. Data Augmentation for Current Q (Update Critic)
        rng, aug_key = jax.random.split(rng)
        aug_obs = self.data_augmentation_fn(aug_key, batch["observations"])
        
        predicted_qs = self.forward_critic(
            aug_obs, batch["actions"], rng=rng, grad_params=params
        )
        
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": predicted_qs.mean(),
            "target_q_mean": target_qs.mean(),
        }

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        # Data Augmentation for Actor Update
        rng, aug_key = jax.random.split(rng)
        aug_obs = self.data_augmentation_fn(aug_key, batch["observations"])
        
        # Forward Policy
        action_dist = self.forward_policy(
            aug_obs, rng=rng, grad_params=params
        )
        actions = action_dist.mode() # Deterministic action
        
        # Forward Critic (using current critic parameters, frozen)
        predicted_qs = self.forward_critic(
            aug_obs,
            actions,
            rng=rng,
        )
        
        q_val = predicted_qs[0] 
        actor_loss = -jnp.mean(q_val)
        
        return actor_loss, {"actor_loss": actor_loss}

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
        }
    
    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        stddev: float = 0.0,
        **kwargs,
    ) -> jnp.ndarray:
        
        dist = self.forward_policy(observations, rng=seed, train=False)
        action = dist.mode()
        
        if not argmax:
            if seed is not None:
                noise = jax.random.normal(seed, shape=action.shape) * stddev
                action = jnp.clip(action + noise, -1.0, 1.0)
            
        return action

