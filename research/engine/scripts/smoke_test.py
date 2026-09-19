"""Quick T1 engine smoke test: qdax-v2 brax env + QDGA for a handful of gens."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from need_ne.brax_conf import get_env_conf, make_centroids
from need_ne.engine import QDGA, QDGAConfig, make_scoring_fn
from need_ne.policy import PolicySpec


def main():
    env_name = sys.argv[1] if len(sys.argv) > 1 else "hopper"
    conf = get_env_conf(env_name)
    env = __import__("qdax.tasks.brax.v2", fromlist=["create"]).create(
        conf.env_name, episode_length=conf.episode_length, backend="generalized"
    )
    spec = PolicySpec(
        input_size=env.observation_size,
        output_size=env.action_size,
        hidden_sizes=[16, 16],
        action_bound=1.0,
    )
    conf = get_env_conf(env_name)
    scoring_fn = make_scoring_fn(
        env, spec,
        __import__("qdax.tasks.brax.v2", fromlist=["descriptor_extractor"]).descriptor_extractor[conf.env_name],
        conf.episode_length,
    )
    algo = QDGA(
        env,
        QDGAConfig(env_conf=conf, policy_spec=spec, pop_size=32, elite_frac=0.15, k_nn=3),
        scoring_fn,
        make_centroids(conf),
        jax.random.PRNGKey(0),
    )
    result, metrics = algo.run(5)
    print("smoke ok: env", env_name)
    for k, v in metrics.items():
        print(" ", k, [float(x) for x in v][:5])


if __name__ == "__main__":
    main()