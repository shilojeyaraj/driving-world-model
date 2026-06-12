"""Behavior (actor-critic) training in imagination -- skeleton.

The policy never touches the env here: it rolls the RSSM forward with rssm.imagine(), the
critic scores imagined states, and you optimize lambda-returns. Fill in once the world model
trains. See models/actor_critic.py.
"""
def train_behavior(cfg):
    # TODO:
    #   1. sample real states from the buffer, encode -> latent (the imagination start points)
    #   2. rssm.imagine() forward cfg.imagine_horizon steps using actor's actions
    #   3. decode predicted reward/continue; compute lambda-returns
    #   4. actor loss = -returns ; critic loss = value regression to returns
    raise NotImplementedError
