"""Closed-loop eval: run the trained actor in the env and measure DRIVING, not prediction.

Concept:  The second eval axis. A model can ace open-loop prediction and still drive badly.
Question: If open-loop looks great but closed-loop fails, what are the likely causes?

Metrics: route completion %, collisions/km, lane-keeping error, interventions/km.
Baselines to beat: random, the data-collection policy, behavior-cloning (no world model).
"""
def closed_loop_eval(actor, world_model, env, episodes=10):
    # TODO: run episodes; encode obs -> latent -> actor(action); step env; aggregate metrics.
    raise NotImplementedError
