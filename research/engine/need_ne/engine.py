"""JAX-native per-generation engine (T1).

The engine ports the legacy elitist-GA recipe and makes it rigorous:
fitness = mean undiscounted episodic return (reward-offset for QD positivity),
descriptors from the QDax Brax v2 wrappers, novelty-alpha selection against a
MAP-Elites archive, uniform crossover + polynomial mutation, and masked
connectivity restructuring of elites. The whole campaign runs as one
``lax.scan`` so per-generation overhead is negligible.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.mdp_utils import generate_unroll
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from .brax_conf import EnvConf
from .operators import (
    gaussian_mutation,
    init_population,
    mask_restructure,
    polynomial_mutation,
    take_individuals,
    tournament_select,
    uniform_crossover,
)
from .policy import PolicySpec


def _get_mask_from_transitions(data: QDTransition) -> jnp.ndarray:
    """Mask with 1 from the first done step of each rollout onward."""
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)
    return mask.astype(jnp.float32)


def make_scoring_fn(
    env,
    spec: PolicySpec,
    descriptor_fn: Callable,
    episode_length: int,
    stochastic: bool = False,
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]:
    """Build the (pop, key) -> (fitness, descriptor, extra) scoring function.

    ``forward`` from policy.py replaces QDax's default flax MLP so our
    masked-genome genotype works. ``stochastic=True`` resets each rollout with a
    fresh key (rigor); the default uses one fixed init state per run.
    """
    from .policy import forward

    def _play_step(env_state, genome, key):
        actions = forward(genome, env_state.obs, spec)
        next_env_state = env.step(env_state, actions)
        state_desc = env_state.info["state_descriptor"]
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=next_env_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_env_state.info["state_descriptor"],
        )
        return next_env_state, genome, key, transition

    play_reset_fn = env.reset if stochastic else (lambda _key: env.reset(jax.random.PRNGKey(0)))
    ep_length = episode_length

    def scoring_fn(policies: Genotype, key: RNGKey):
        batch = jax.tree.leaves(policies)[0].shape[0]
        key, subkey = jax.random.split(key)
        reset_keys = jax.random.split(subkey, batch)
        init_states = jax.vmap(play_reset_fn)(reset_keys)

        unroll_fn = partial(
            generate_unroll, episode_length=ep_length, play_step_fn=_play_step
        )
        rollout_keys = jax.random.split(key, batch)
        _, data = jax.vmap(unroll_fn)(init_states, policies, rollout_keys)

        mask = _get_mask_from_transitions(data)
        fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
        descriptors = descriptor_fn(data, mask)
        return fitnesses, descriptors, {"transitions": data}

    return scoring_fn


def _rank_norm(values: jnp.ndarray) -> jnp.ndarray:
    """Rank-normalize a batch into [0, 1], higher rank = better."""
    n = values.shape[0]
    if n <= 1:
        return jnp.zeros_like(values)
    order = jnp.argsort(values)
    ranks = jnp.empty_like(order, dtype=jnp.int32)
    ranks = ranks.at[order].set(jnp.arange(n))
    return ranks.astype(jnp.float32) / (n - 1)


def novelty_alpha_scores(
    fitnesses: jnp.ndarray,
    novelty: jnp.ndarray,
    alpha: float,
    use_novelty: bool,
) -> jnp.ndarray:
    if not use_novelty:
        return _rank_norm(fitnesses)
    return alpha * _rank_norm(fitnesses) + (1.0 - alpha) * _rank_norm(novelty)


def compute_novelty(
    descriptors: jnp.ndarray,
    archive: MapElitesRepertoire,
    k_nn: int = 5,
) -> jnp.ndarray:
    """Mean distance of each candidate to its k-nn filled archive cells."""
    arch_desc = archive.descriptors
    filled = (jnp.isfinite(archive.fitnesses) & (archive.fitnesses > -1e30)).reshape(-1)
    d2 = jnp.sum((descriptors[:, None, :] - arch_desc[None, :, :]) ** 2, axis=-1)
    d2 = jnp.where(filled[None, :], d2, jnp.inf)
    k = min(k_nn, int(arch_desc.shape[0]))
    return jnp.sqrt(jnp.sort(d2, axis=1)[:, :k]).mean(axis=1)


@dataclass
class QDGAConfig:
    env_conf: EnvConf
    policy_spec: PolicySpec
    pop_size: int = 128
    elite_frac: float = 0.15
    alpha: float = 0.7
    k_nn: int = 5
    mutation: str = "polynomial"
    weight_rate: float = 0.1
    bias_rate: float = 0.1
    mask_mut_rate: float = 0.025
    mut_power: float = 0.5
    eta: float = 20.0
    reorganize_prob: float = 1.0
    tournament_k: int = 4
    use_qd: bool = True
    use_novelty: bool = True
    use_restructuring: bool = True
    stochastic_train: bool = False


class QDGA:
    """Elitist GA with novelty-alpha selection + MAP-Elites archive."""

    def __init__(
        self,
        env,
        config: QDGAConfig,
        scoring_fn: Callable,
        centroids: jnp.ndarray,
        key: RNGKey,
    ):
        self.env = env
        self.config = config
        self.scoring_fn = scoring_fn
        self.centroids = centroids
        self.key = key

        self.pop_size = config.pop_size
        self.elite_count = max(1, int(round(config.pop_size * config.elite_frac)))
        self.restructured_count = self.elite_count if config.use_restructuring else 0
        self.child_count = config.pop_size - self.elite_count - self.restructured_count
        if self.child_count % 2 != 0:
            self.elite_count += 1
            self.child_count -= 1
        if self.child_count < 2:
            raise ValueError(
                "pop layout leaves too few offspring slots; raise pop_size or lower elite_frac"
            )

    def _step(self):
        conf = self.config
        spec = conf.policy_spec
        elite_count = self.elite_count
        child_count = self.child_count
        pairs = child_count // 2
        # rebuild any static sizes the closure needs
        del child_count

        @jax.jit
        def step(carry):
            pop, archive, key, step_idx = carry
            key, subkey = jax.random.split(key)

            fitnesses, descriptors, _ = self.scoring_fn(pop, subkey)

            if conf.use_qd:
                archive = archive.add(pop, descriptors, fitnesses)

            if conf.use_novelty:
                novelty = compute_novelty(descriptors, archive, conf.k_nn)
            else:
                novelty = jnp.zeros_like(fitnesses)

            scores = novelty_alpha_scores(fitnesses, novelty, conf.alpha, conf.use_novelty)

            order = jnp.argsort(-scores, stable=True)
            elite_idx = order[:elite_count]
            elites = take_individuals(pop, elite_idx)

            key, subkey = jax.random.split(key)
            parent_idx = tournament_select(subkey, scores, conf.tournament_k, 2 * pairs)
            p1 = take_individuals(pop, parent_idx[:pairs])
            p2 = take_individuals(pop, parent_idx[pairs : 2 * pairs])

            key, subkey = jax.random.split(key)
            c1, c2 = uniform_crossover(subkey, p1, p2, spec)
            children = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), c1, c2)

            key, subkey = jax.random.split(key)
            if conf.mutation == "polynomial":
                children = polynomial_mutation(
                    subkey, children, spec,
                    conf.weight_rate, conf.bias_rate, conf.mask_mut_rate,
                    conf.mut_power, conf.eta,
                )
            else:
                children = gaussian_mutation(
                    subkey, children, spec,
                    conf.weight_rate, conf.bias_rate, conf.mut_power, conf.mut_power,
                )

            parts = [elites]
            if conf.use_restructuring:
                key, subkey = jax.random.split(key)
                clones = mask_restructure(subkey, elites, spec, conf.reorganize_prob)
                parts.append(clones)
            parts.append(children)
            new_pop = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)

            best = jnp.max(fitnesses)
            mean = jnp.mean(fitnesses)
            if conf.use_qd:
                filled = jnp.isfinite(archive.fitnesses) & (archive.fitnesses > -1e30)
                qd_score = jnp.sum(jnp.where(filled, archive.fitnesses, 0.0))
                coverage = jnp.mean(filled)
                qd_best = jnp.max(jnp.where(filled, archive.fitnesses, -jnp.inf))
            else:
                qd_score, coverage, qd_best = jnp.nan, jnp.nan, best
            novelty_mean = jnp.mean(novelty)

            metrics = {
                "step": step_idx,
                "best_fitness": best,
                "mean_fitness": mean,
                "qd_best_fitness": qd_best,
                "qd_score": qd_score,
                "coverage": coverage,
                "novelty_mean": novelty_mean,
            }
            return (new_pop, archive, key, step_idx + 1), metrics

        return step

    def run(self, generations: int):
        conf = self.config
        key, subkey = jax.random.split(self.key)
        pop = init_population(subkey, conf.policy_spec, self.pop_size)

        key, skey = jax.random.split(key)
        fitnesses, descriptors, _ = self.scoring_fn(pop, skey)
        archive = MapElitesRepertoire.init(
            genotypes=pop,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=self.centroids,
        )

        step_fn = self._step()
        init_carry = (pop, archive, key, jnp.asarray(0, dtype=jnp.int32))
        final_carry, metrics = jax.lax.scan(
            lambda c, _x: step_fn(c), init_carry, xs=None, length=generations
        )
        final_pop, final_archive, _, _ = final_carry
        return {"genotypes": final_pop, "archive": final_archive}, metrics