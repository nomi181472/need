"""FAST operators: batched, jit-friendly genetic operators (T1).

The genome is a pytree
    {"weights":  list of [B, out, in] matrices,
     "biases":   list of [B, out] vectors,
     "masks":    list of [B, hidden] binary vectors (hidden layers only)}

All operators are elementwise on the batch dimension so they compose inside a
jit-per-generation loop.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .policy import PolicySpec, init_genome


def init_population(key: jax.Array, spec: PolicySpec, pop_size: int) -> Dict:
    """Sample pop_size genomes, batched along axis 0."""
    keys = jax.random.split(key, pop_size)
    pop = jax.vmap(init_genome, in_axes=(0, None))(keys, spec)
    return pop


def take_individuals(pop: Dict, indices: jax.Array) -> Dict:
    """Gather batch elements by index (stable across leaves)."""
    return jax.tree.map(lambda x: x[indices], pop)


def concat_pops(*pops: Dict) -> Dict:
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *pops)


@partial(jax.jit, static_argnames=("spec",))
def uniform_crossover(
    key: jax.Array, p1: Dict, p2: Dict, spec: PolicySpec
) -> Tuple[Dict, Dict]:
    """Uniform crossover producing two children (binomial per-matrix swap)."""
    k1, k2 = jax.random.split(key)

    def _pick(k, x, y):
        c = jax.random.uniform(k, x.shape, x.dtype)
        return jnp.where(c < 0.5, x, y)

    c1 = jax.tree.map(lambda x, y: _pick(k1, x, y), p1, p2)
    c2 = jax.tree.map(lambda x, y: _pick(k2, x, y), p2, p1)
    return c1, c2


@partial(jax.jit, static_argnames=("spec",))
def polynomial_mutation(
    key: jax.Array,
    genome: Dict,
    spec: PolicySpec,
    weight_rate: float = 0.1,
    bias_rate: float = 0.1,
    mask_rate: float = 0.025,
    mut_power: float = 0.5,
    eta: float = 20.0,
    use_masks: bool = True,
) -> Dict:
    """Polynomial mutation (Deb) on weights/biases; binary mask flips on hidden layers."""
    subkeys = jax.random.split(key, len(genome["weights"]) + len(genome["biases"]) + len(genome["masks"]))

    def _poly_mut(k, x, lo, hi, rate):
        u = jax.random.uniform(k, x.shape, x.dtype)
        apply = u < rate
        norm = (x - lo) / (hi - lo)
        uu = jax.random.uniform(k, norm.shape, norm.dtype)
        delta = jnp.where(
            uu <= 0.5,
            (2.0 * uu) ** (1.0 / (eta + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - uu)) ** (1.0 / (eta + 1.0)),
        )
        new_norm = jnp.clip(norm + delta * mut_power, 0.0, 1.0)
        new_x = lo + new_norm * (hi - lo)
        return jnp.where(apply, new_x, x)

    new_weights = [
        _poly_mut(
            subkeys[i],
            w,
            spec.weight_lo,
            spec.weight_hi,
            weight_rate,
        )
        for i, w in enumerate(genome["weights"])
    ]
    n_w = len(genome["weights"])
    new_biases = [
        _poly_mut(
            subkeys[n_w + i],
            b,
            spec.bias_lo,
            spec.bias_hi,
            bias_rate,
        )
        for i, b in enumerate(genome["biases"])
    ]

    n_b = len(genome["biases"])
    new_masks = []
    for i, m in enumerate(genome["masks"]):
        if not use_masks:
            new_masks.append(m)
            continue
        k = subkeys[n_w + n_b + i]
        flip = jax.random.uniform(k, m.shape, m.dtype) < mask_rate
        new_masks.append(jnp.where(flip, 1.0 - m, m))

    return {"weights": new_weights, "biases": new_biases, "masks": new_masks}


@partial(jax.jit, static_argnames=("spec",))
def gaussian_mutation(
    key: jax.Array,
    genome: Dict,
    spec: PolicySpec,
    weight_rate: float = 0.1,
    bias_rate: float = 0.1,
    weight_power: float = 0.1,
    bias_power: float = 0.1,
    use_masks: bool = True,
) -> Dict:
    """Gaussian mutation on weights/biases with clamping to genome bounds."""
    def _gauss_mut(k, x, lo, hi, rate, power):
        shape = x.shape
        u = jax.random.uniform(k, shape, x.dtype)
        noise = jax.random.normal(k, shape, x.dtype) * power
        return jnp.clip(jnp.where(u < rate, x + noise, x), lo, hi)

    subkeys = jax.random.split(key, len(genome["weights"]) + len(genome["biases"]))

    new_weights = [
        _gauss_mut(
            subkeys[i],
            w,
            spec.weight_lo,
            spec.weight_hi,
            weight_rate,
            weight_power,
        )
        for i, w in enumerate(genome["weights"])
    ]
    n_w = len(genome["weights"])
    new_biases = [
        _gauss_mut(
            subkeys[n_w + i],
            b,
            spec.bias_lo,
            spec.bias_hi,
            bias_rate,
            bias_power,
        )
        for i, b in enumerate(genome["biases"])
    ]
    return {"weights": new_weights, "biases": new_biases, "masks": genome["masks"]}


@partial(jax.jit, static_argnames=("spec", "reorganize_prob"))
def mask_restructure(
    key: jax.Array,
    genome: Dict,
    spec: PolicySpec,
    reorganize_prob: float = 1.0,
) -> Dict:
    """Masked connectivity restructuring (legacy ``treatment()``).

    For each hidden layer, underutilized pathways (mask==0) can be re-enabled and
    active pathways pruned, producing connectivity-different clones of elites.
    ``reorganize_prob`` gates how aggressively masks are re-drawn.
    """
    if spec.num_hidden == 0:
        return genome

    keys = jax.random.split(key, len(genome["masks"]) + 1)
    new_masks = []
    for i, m in enumerate(genome["masks"]):
        flip = jax.random.uniform(keys[i], m.shape, m.dtype) < reorganize_prob * 0.5
        new_masks.append(jnp.where(flip, jnp.abs(m - 1.0), m))

    # re-tune weights slightly so restructured clones are not identical to elites
    noise_key = keys[-1]
    noise = jax.random.normal(noise_key, genome["weights"][0].shape, genome["weights"][0].dtype) * 0.05
    new_weights = [jnp.clip(genome["weights"][0] + noise, spec.weight_lo, spec.weight_hi)] + list(genome["weights"])[1:]

    return {"weights": new_weights, "biases": genome["biases"], "masks": new_masks}


@partial(jax.jit, static_argnames=("k", "n_out"))
def tournament_select(
    key: jax.Array,
    scores: jax.Array,
    k: int,
    n_out: int,
) -> jax.Array:
    """Tournament selection returning ``n_out`` indices into ``scores``."""
    pop_n = scores.shape[0]
    rand = jax.random.randint(key, (n_out, k), 0, pop_n)
    noise = jax.random.uniform(key, (n_out, k)) * 1e-6
    winners = jnp.argmax(scores[rand] + noise, axis=1)
    return winners