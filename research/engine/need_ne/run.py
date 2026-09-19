"""Campaign runner (T1).

Runs the QDGA engine for one or more seeds on a Brax headline env, writing a
per-generation metrics CSV, the best/pop genotypes, and a final evaluation
summary (mean/SD over eval seeds, single vs ensemble).

Example:
    python -m need_ne.run --env hopper --seeds 42 --generations 30 --pop-size 64
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .brax_conf import get_env_conf, make_centroids
from .engine import QDGA, QDGAConfig, make_scoring_fn
from .evaluate import evaluate_genotype
from .ensemble import ensemble_forward_batch
from .policy import PolicySpec


def _extractor(env_name: str):
    from qdax.tasks.brax import v2 as qdax_brax_v2

    return qdax_brax_v2.descriptor_extractor[env_name]


def make_env(env_name: str):
    from qdax.tasks.brax import v2 as qdax_brax_v2

    conf = get_env_conf(env_name)
    return qdax_brax_v2.create(conf.env_name, episode_length=conf.episode_length, backend="generalized")


def _policy_spec(env, env_name: str) -> PolicySpec:
    conf = get_env_conf(env_name)
    return PolicySpec(
        input_size=env.observation_size,
        output_size=env.action_size,
        hidden_sizes=[64, 64],
        action_bound=1.0,
    )


def run_seed(args, env, env_name: str, seed: int, out_dir: Path) -> Path:
    conf = get_env_conf(env_name)
    spec = _policy_spec(env, env_name)
    key = jax.random.PRNGKey(seed)

    scoring_fn = make_scoring_fn(
        env,
        spec,
        _extractor(conf.env_name),
        conf.episode_length,
        stochastic=False,
    )

    centroids = make_centroids(conf)
    config = QDGAConfig(
        env_conf=conf,
        policy_spec=spec,
        pop_size=args.pop_size,
        elite_frac=args.elite_frac,
        alpha=args.alpha,
        k_nn=args.k_nn,
        mutation=args.mutation,
        mask_mut_rate=args.mask_mut_rate,
        mut_power=args.mut_power,
        use_qd=True,
        use_novelty=args.use_novelty,
        use_restructuring=args.use_restructuring,
    )
    algo = QDGA(env, config, scoring_fn, centroids, key)

    t0 = time.time()
    result, metrics = algo.run(args.generations)
    wall = time.time() - t0

    per_gen = {
        "gen": np.asarray(metrics["step"]),
        "best_fitness": np.asarray(metrics["best_fitness"]),
        "mean_fitness": np.asarray(metrics["mean_fitness"]),
        "qd_best_fitness": np.asarray(metrics["qd_best_fitness"]),
        "qd_score": np.asarray(metrics["qd_score"]),
        "coverage": np.asarray(metrics["coverage"]),
        "novelty_mean": np.asarray(metrics["novelty_mean"]),
    }
    steps = (np.arange(args.generations) + 1) * (args.pop_size * conf.episode_length)
    per_gen["samples"] = steps

    csv_path = out_dir / f"per_gen_{env_name}_seed{seed}.csv"
    import pandas as pd

    pd.DataFrame(per_gen).to_csv(csv_path, index=False)

    archive = result["archive"]
    final_pop = result["genotypes"]
    _save_pytree(out_dir / f"final_pop_{env_name}_seed{seed}.npz", final_pop)
    best_genome = jax.tree.map(lambda x: x[0], final_pop)
    _save_pytree(out_dir / f"best_genome_{env_name}_seed{seed}.npz", best_genome)
    best_arch_idx = int(np.argmax(np.where(np.isfinite(archive.fitnesses), archive.fitnesses, -np.inf)))
    best_genome = jax.tree.map(lambda x: x[best_arch_idx], archive.genotypes)

    eval_best = evaluate_genotype(
        env, best_genome, spec, seed=seed + 1000, episode_length=conf.episode_length, n_envs=args.eval_envs
    )

    eval_ens = evaluate_ensemble(env, archive, spec, env_name, seed=seed + 2000, n_envs=args.eval_envs)

    summary = {
        "env": env_name,
        "seed": seed,
        "generations": args.generations,
        "pop_size": config.pop_size,
        "wall_clock_seconds": wall,
        "samples_total": int(per_gen["samples"][-1]),
        "best_arch_idx": best_arch_idx,
        "eval_single": eval_best,
        "eval_ensemble": eval_ens,
    }
    with open(out_dir / f"eval_{env_name}_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return csv_path


def evaluate_ensemble(env, archive, spec, env_name, seed, n_envs):
    from .evaluate import rollout_policy

    conf = get_env_conf(env_name)

    def act(o):
        return ensemble_forward_batch(archive, 5, o, spec)

    returns = rollout_policy(env, act, jax.random.PRNGKey(seed), conf.episode_length, n_envs)
    returns = np.asarray(returns)
    return {"mean": float(returns.mean()), "std": float(returns.std()), "all": returns}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError


def _save_pytree(path: Path, tree):
    """Persist a genome pytree as an npz of named leaves."""
    leaves = jax.tree.leaves(tree)
    np.savez(
        path,
        **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)},
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=["halfcheetah", "hopper", "walker2d", "ant", "humanoid"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--pop-size", type=int, default=128)
    p.add_argument("--elite-frac", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--k-nn", type=int, default=5)
    p.add_argument("--mutation", default="polynomial", choices=["polynomial", "gaussian"])
    p.add_argument("--mask-mut-rate", type=float, default=0.025)
    p.add_argument("--mut-power", type=float, default=0.5)
    p.add_argument("--use-novelty", action="store_true", default=True)
    p.add_argument("--use-restructuring", action="store_true", default=True)
    p.add_argument("--eval-envs", type=int, default=10)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    env_aliases = ["halfcheetah", "hopper", "walker2d", "ant", "humanoid"]
    if args.env not in env_aliases:
        raise SystemExit(f"choose one of {env_aliases}")

    out_dir = Path(args.out_dir) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env)
    for seed in args.seeds:
        run_seed(args, env, args.env, seed, out_dir)
        print(f"seed {seed} done")

    if len(args.seeds) > 1:
        summarize(args.env, args.seeds, out_dir)


def summarize(env_name, seeds, out_dir: Path):
    import pandas as pd

    frames = [pd.read_csv(out_dir / f"per_gen_{env_name}_seed{s}.csv") for s in seeds]
    big = pd.concat(frames, keys=seeds, names=["seed", "row"])
    grouped = big.groupby(level="row")
    stats = pd.DataFrame(
        {
            "gen": grouped["gen"].first(),
            "best_fitness_mean": grouped["best_fitness"].mean(),
            "best_fitness_std": grouped["best_fitness"].std(),
            "qd_best_fitness_mean": grouped["qd_best_fitness"].mean(),
            "coverage_mean": grouped["coverage"].mean(),
            "qd_score_mean": grouped["qd_score"].mean(),
        }
    )
    stats.to_csv(out_dir / f"stats_{env_name}.csv", index=False)
    print(stats.tail(3).to_string())


if __name__ == "__main__":
    main()