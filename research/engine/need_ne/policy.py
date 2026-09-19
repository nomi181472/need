"""T1 engine: masked MLP policy for Brax.

Genome is a jit-friendly pytree of flat arrays:
    weights: list of [out, in] weight matrices (hidden + output layers)
    biases:  list of [out] bias vectors (same number as weights)
    masks:   list of [out] mask vectors for HIDDEN layers only
             (masks[i] scales the outgoing weights of hidden layer i)

Forward pass applies mask scaling to hidden outgoing weights -> TanH hidden
activations -> TanH output scaled by the environment action bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import List

import jax
import jax.numpy as jnp


@dataclass(eq=False)
class PolicySpec:
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    action_bound: float = 1.0
    init_mean: float = 0.0
    init_std: float = 0.3
    weight_lo: float = -3.0
    weight_hi: float = 3.0
    bias_lo: float = -3.0
    bias_hi: float = 3.0

    @property
    def num_hidden(self) -> int:
        return len(self.hidden_sizes)

    @property
    def num_layers(self) -> int:
        return len(self.hidden_sizes) + 1  # hidden layers + output layer


def _genome_shapes(spec: PolicySpec) -> List[tuple]:
    """Return [(out, in)] shapes for each weight matrix."""
    sizes = [spec.input_size] + list(spec.hidden_sizes) + [spec.output_size]
    return [(sizes[i + 1], sizes[i]) for i in range(len(sizes) - 1)]


def init_genome(key: jax.Array, spec: PolicySpec) -> dict:
    """Create a random bounded genome for one policy."""
    weights, biases = [], []
    for out_dims, in_dims in _genome_shapes(spec):
        k1, k2, key = jax.random.split(key, 3)
        w = jax.random.normal(k1, (out_dims, in_dims)) * spec.init_std
        b = jax.random.normal(k2, (out_dims,)) * spec.init_std
        weights.append(jnp.clip(w, spec.weight_lo, spec.weight_hi))
        biases.append(jnp.clip(b, spec.bias_lo, spec.bias_hi))

    masks = []
    for hidden_size in spec.hidden_sizes:
        k, key = jax.random.split(key)
        masks.append(jnp.ones((hidden_size,), dtype=jnp.float32) * (jax.random.uniform(k, ()) > 0.0))

    return {"weights": weights, "biases": biases, "masks": masks}


@partial(jax.jit, static_argnames=("spec", "is_discrete"))
def forward(genome: dict, obs: jax.Array, spec: PolicySpec, is_discrete: bool = False) -> jax.Array:
    """Vectorized forward pass. obs: (..., input_size)."""
    x = obs
    weights, biases, masks = genome["weights"], genome["biases"], genome["masks"]
    n_hidden = spec.num_hidden
    for i in range(n_hidden):
        w = weights[i]
        if i < len(masks):
            w = w * masks[i][:, None]  # scale outgoing rows of the hidden layer
        x = jnp.tanh(jnp.matmul(x, w.T) + biases[i])
    out = jnp.tanh(jnp.matmul(x, weights[-1].T) + biases[-1])
    if is_discrete:
        return out  # logits
    return out * spec.action_bound


def count_parameters(spec: PolicySpec) -> int:
    n = 0
    for out_dims, in_dims in _genome_shapes(spec):
        n += out_dims * in_dims + out_dims
    return n