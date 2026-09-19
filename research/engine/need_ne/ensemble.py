"""Ensemble policies (T1).

The legacy system voted/averaged over the top-k archive members at test time.
For jit-friendliness we average the bounded action outputs of the k best
archive genotypes; the mean stays inside the action bound.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

from .policy import PolicySpec, forward


def top_k_genotypes(
    archive: MapElitesRepertoire, k: int
) -> jax.Array:
    """Indices of the k best filled archive cells (descending fitness)."""
    filled = (jnp.isfinite(archive.fitnesses) & (archive.fitnesses > -1e30)).reshape(-1)
    score = jnp.where(filled, archive.fitnesses.reshape(-1), -jnp.inf)
    order = jnp.argsort(-score, stable=True)
    return order[:k]


def ensemble_forward_batch(
    archive: MapElitesRepertoire,
    k: int,
    obs: jax.Array,
    spec: PolicySpec,
) -> jax.Array:
    """Mean of the k best archive policies' actions for a batch of obs."""
    idx = top_k_genotypes(archive, k)
    genotypes = jax.tree.map(lambda x: x[idx], archive.genotypes)
    actions = jax.vmap(lambda g, o: forward(g, o, spec), in_axes=(0, None))(genotypes, obs)
    return jnp.mean(actions, axis=0)