"""need_ne: QD-enhanced neuroevolution engine (T1).

JAX-native, jit-per-generation engine built on QDax's Brax v2 task components.
Genomes are masked MLP pytrees (see policy.py). The engine ports the legacy
elitist-GA operators (uniform crossover, polynomial/gaussian mutation, masked
connectivity restructuring) and adds a MAP-Elites archive, novelty-alpha
selection and ensemble evaluation.
"""