import pytest
import numpy as np
from pettingzoo.test import parallel_seed_test
from DSSE.environment.constants import Actions
from DSSE.environment.entities.person import Person
from DSSE.tests.test_env import init_drone_swarm_search


def rollout(env, seed, steps=30, reset_seed=True):
    """Run a fixed action sequence and record everything stochastic."""
    observations, _ = env.reset(seed=seed if reset_seed else None)
    trace = [sorted(person.get_position() for person in env.get_persons())]
    for step in range(steps):
        if not env.agents:
            break
        action = list(Actions)[step % len(Actions)].value
        observations, rewards, _, _, _ = env.step(
            {agent: action for agent in env.agents}
        )
        trace.append(
            (
                sorted(person.get_position() for person in env.get_persons()),
                sorted(rewards.items()),
                [obs[1].sum() for obs in observations.values()],
            )
        )
    return trace


def test_parallel_seed_test():
    parallel_seed_test(
        lambda: init_drone_swarm_search(
            grid_size=10, drone_amount=2, person_amount=3, timestep_limit=50
        )
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_same_seed_gives_same_episode(seed):
    env_a = init_drone_swarm_search(person_amount=4, drone_amount=2)
    env_b = init_drone_swarm_search(person_amount=4, drone_amount=2)
    assert rollout(env_a, seed) == rollout(env_b, seed)


def test_same_seed_gives_same_episode_on_the_same_instance():
    env = init_drone_swarm_search(person_amount=4, drone_amount=2)
    first = rollout(env, seed=7)
    rollout(env, seed=99)
    assert rollout(env, seed=7) == first


def test_different_seeds_give_different_episodes():
    env = init_drone_swarm_search(person_amount=4, drone_amount=2)
    traces = {str(rollout(env, seed)) for seed in range(5)}
    assert len(traces) > 1


def test_unseeded_reset_continues_the_seeded_stream():
    env_a = init_drone_swarm_search(person_amount=4, drone_amount=2)
    env_b = init_drone_swarm_search(person_amount=4, drone_amount=2)
    first_a = rollout(env_a, seed=3)
    rollout(env_b, seed=3)
    second_a = rollout(env_a, seed=None, reset_seed=False)
    second_b = rollout(env_b, seed=None, reset_seed=False)
    # Reproducible across instances, but not a replay of the first episode.
    assert second_a == second_b
    assert second_a != first_a


def test_np_random_seed_is_exposed():
    env = init_drone_swarm_search()
    env.reset(seed=5)
    assert env.np_random_seed == 5


def test_person_without_rng_still_moves():
    person = Person(index=0, initial_position=(5, 5), grid_size=10)
    person.calculate_movement_vector((1.0, 1.0))
    movement_map = np.full((3, 3), 1 / 9)
    for _ in range(20):
        person.step(movement_map.copy())
    assert 0 <= person.x < 10 and 0 <= person.y < 10
