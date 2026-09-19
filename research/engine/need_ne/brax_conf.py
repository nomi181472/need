"""Task configuration for the Brax headline environments (T1).

Environment names use qdax's Brax v2 namespaces (_uni = feet-contact 1D
descriptor, _omni = final-xy 2D descriptor). Values follow qdax 0.5.0's brax
conventions (reward_offset keeps fitnesses positive for QD scoring).
"""
from __future__ import annotations

from dataclasses import dataclass

from qdax.tasks.brax import v2 as qdax_brax_v2


@dataclass(frozen=True)
class EnvConf:
    env_name: str
    episode_length: int
    descriptor_dim: int
    grid_size: int
    minval: float
    maxval: float


_ENVS = {
    "halfcheetah": EnvConf("halfcheetah_uni", 500, 1, 1024, 0.0, 1.0),
    "hopper": EnvConf("hopper_uni", 500, 1, 1024, 0.0, 1.0),
    "walker2d": EnvConf("walker2d_uni", 500, 1, 1024, 0.0, 1.0),
    "ant": EnvConf("ant_omni", 500, 2, 1024, -20.0, 20.0),
    "humanoid": EnvConf("humanoid_omni", 1000, 2, 1024, -20.0, 20.0),
}

_SUPPORTED = frozenset(_ENVS)


def get_env_conf(name: str) -> EnvConf:
    if name not in _SUPPORTED:
        raise ValueError(f"unsupported env {name!r}; pick one of {sorted(_SUPPORTED)}")
    return _ENVS[name]


def make_centroids(conf: EnvConf) -> "jax.Array":
    """Grid centroids (32x32 = 1024 cells for 2D, 1024 points for 1D)."""
    import jax.numpy as jnp

    if conf.descriptor_dim == 1:
        return jnp.linspace(conf.minval, conf.maxval, conf.grid_size).reshape(-1, 1)
    n = int(round(conf.grid_size ** 0.5))
    axes = [
        jnp.linspace(conf.minval, conf.maxval, n),
        jnp.linspace(conf.minval, conf.maxval, n),
    ]
    g1, g2 = jnp.meshgrid(*axes)
    return jnp.stack([g1.ravel(), g2.ravel()], axis=1)


def descriptor_extractor(env_name: str):
    """Returns qdax's descriptor extractor callable for the env."""
    return qdax_brax_v2.descriptor_extractor[env_name]


def reward_offset(env_name: str) -> float:
    return qdax_brax_v2.reward_offset.get(env_name, 0.0)