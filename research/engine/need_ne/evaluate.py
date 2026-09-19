"""Policy evaluation utilities (T1).

Rigorous evaluation: rollout a (possibly ensembled) policy over R repeat seeds
in a fresh stochastic environment and report mean/SD/CI of the return.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from .policy import PolicySpec, forward


def rollout_policy(
    env,
    act_fn: Callable,
    key: jax.Array,
    episode_length: int,
    n_envs: int,
) -> jax.Array:
    """Roll out ``act_fn(obs) -> actions`` for n_envs stochastic episodes.

    Returns per-episode undiscounted return (shape [n_envs]).
    """

    def _step(carry, _):
        env_state, total, key = carry
        actions = act_fn(env_state.obs)
        env_state = env.step(env_state, actions)
        total += env_state.reward * (1.0 - env_state.done)
        return (env_state, total, key), None

    def _one(k):
        s = env.reset(k)
        (_, total, _), _ = jax.lax.scan(
            _step, (s, jnp.zeros_like(s.reward), k), None, length=episode_length
        )
        return total

    return jax.vmap(_one)(jax.random.split(key, n_envs))


def evaluate_genotype(
    env,
    genome,
    spec: PolicySpec,
    seed: int = 0,
    episode_length: int = 1000,
    n_envs: int = 10,
) -> dict:
    """Evaluate a single genome: mean/SD return over n_envs seeds."""
    import numpy as np

    act_fn = jax.jit(partial(forward, spec=spec))
    act = partial(act_fn, genome)
    returns = rollout_policy(env, act, jax.random.PRNGKey(seed), episode_length, n_envs)
    returns = np.asarray(returns)
    return {"mean": float(returns.mean()), "std": float(returns.std()), "all": returns}