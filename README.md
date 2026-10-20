# MARL_SPI

MARL_SPI is a Python/PyTorch library of multi-agent reinforcement learning components for autonomous vehicle control in a three-lane road-merging setting. It combines independent learners with optional driving priors, feasibility masks, distance-based risk weighting, and context-conditioned network initialization.

The repository contains the learning and control building blocks. To run traffic experiments, supply a simulator, observation adapter, training loop, and prior data. A simulator integration, command-line training script, datasets, and pretrained checkpoints.

## Features

- **Independent learners:** proximal policy optimization (IPPO), advantage actor-critic (IA2C), and deep Q-learning (IDQN).
- **Prior-guided learning:** empirical priors, IDM/MOBIL model-derived priors, hybrid prior fusion, and adaptive KL regularization.
- **Action selection:** hard feasibility masks plus optional prior- and risk-based soft weighting.
- **Contextual initialization:** a context encoder, actor/critic hypernetworks, and feature modulation.
- **Traffic utilities:** local observations, semantic actions, bicycle-model dynamics, geometric clearance, closest approach, and reward components.
- **Behavior cloning:** a supervised actor update from demonstrated actions.

## Setup

Use Python **3.10 or newer**; the source uses Python 3.10 union type syntax. The external runtime dependencies are **NumPy** and **PyTorch**.

```bash
git clone https://github.com/imba-ang/MARLSPI.git
cd MARLSPI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch
```

Run the examples from the repository root so Python can import `marl_spi`. 

## Quick start

This example creates an IPPO agent, selects a feasible action, converts it to a vehicle command, and performs one update using a synthetic transition. 

Save it as `quickstart.py` in the repository root and run `python quickstart.py`.

```python
import torch

from marl_spi import LocalObservation, Neighbor, Role, create_actor_critic_agent
from marl_spi.actions import execute_action
from marl_spi.config import LearningConfig
from marl_spi.policies import feasible_actions

torch.manual_seed(0)

observation = LocalObservation.from_roles(
    lane=2,
    speed=12.0,
    merge_distance=150.0,
    roles={Role.SV: Neighbor(distance=30.0, relative_speed=0.0)},
)
feasible = feasible_actions(
    lane=observation.lane,
    ramp_merge_available=False,
    left_available=True,
    right_available=False,
)
agent = create_actor_critic_agent(
    algorithm="ippo",
    config=LearningConfig(update_epochs=1),
)

decision = agent.act(observation, feasible)
command = execute_action(decision.action, observation.lane, feasible)
print(decision.action.name, command)

# Replace these values with the result of stepping your simulator.
next_observation = LocalObservation.from_roles(
    lane=command.target_lane,
    speed=12.0,
    merge_distance=148.0,
    roles={Role.SV: Neighbor(distance=30.0, relative_speed=0.0)},
)
agent.observe(
    decision,
    reward=1.0,
    next_observation=next_observation,
    terminated=False,
    truncated=True,
)
print(agent.update())
```

Use `algorithm="ia2c"` for IA2C, which requires `update_epochs=1`. Both learners return loss and KL metrics and clear their rollout buffer after updating. For evaluation, use `agent.act(..., deterministic=True)`. 

IPPO and IA2C learn from freshly collected on-policy rollouts. IPPO supports multiple clipped-objective update epochs; IA2C performs one actor-critic update epoch. Choose IDQN when your experiment calls for Q-learning with replay.

IDQN is available through `create_q_agent()`. Its `observe()` method takes `next_feasible` and `terminated`, without a `truncated` argument. Collect at least `PAPER.batch_size` transitions (128 by default) before calling its `update()`, which samples from replay. Without a prior, IDQN uses epsilon-greedy selection. With a prior, it samples a masked distribution over temperature-scaled Q-values.

## Observations and actions

`LocalObservation` stores a lane index (`1`, `2`, or `3`), speed, distance to the merge, and six neighbor slots in this order:

```text
SV, FV, NLV, NPV, RLV, RPV
```

`LocalObservation.from_roles()` fills omitted slots as unavailable. Its `vector()` method returns a 21-element `float32` NumPy array:

```text
[lane, speed, merge_distance,
 SV_distance, SV_relative_speed, SV_available,
 ...,
 RPV_distance, RPV_relative_speed, RPV_available]
```

Unavailable neighbors and neighbors beyond the default perception range contribute three zeros. Features are returned without normalization. `StateAbstraction` uses `Role.SV` as the leader.

All action masks, priors, and risk vectors use the following order:

| Index | Action | Default command |
| --- | --- | --- |
| 0 | `ACCELERATE` | Acceleration of +2 m/s² |
| 1 | `DECELERATE` | Acceleration of −3 m/s² |
| 2 | `MAINTAIN` | Acceleration of 0 m/s² |
| 3 | `LEFT` | Target lane = current lane + 1 |
| 4 | `RIGHT` | Target lane = current lane − 1 |

Lane changes use zero nominal acceleration by default. `execute_action()` returns a `Command(acceleration, target_lane)`; the simulator must implement the target-lane maneuver.

`feasible_actions()` encodes the repository's road layout: lane 1 can merge left when permitted, lane 2 can move left, and lane 3 can move right. The caller supplies lane-availability flags. 

Agent `act()` methods accept masks convertible to a five-element Boolean tensor. At least one action must be feasible.

## Using a prior

Pass a callable `prior(abstract_state)` returning five action probabilities to either agent factory. This enables prior-based soft weighting during action selection and adaptive KL regularization during learning. A prior-enabled `act()` also requires a five-element risk vector with values in `[0, 1]`.

The following continues the quick-start example with a manually specified prior for its observation:

```python
from marl_spi.priors import FrozenPrior, StateAbstraction

abstraction = StateAbstraction()
prior = FrozenPrior({
    abstraction(observation): (0.15, 0.15, 0.40, 0.15, 0.15),
})
spi_agent = create_actor_critic_agent(prior=prior, abstraction=abstraction)

# Illustrative values; compute action-specific risks from your simulator.
risk = torch.tensor([0.1, 0.0, 0.0, 0.3, 0.3])
spi_decision = spi_agent.act(observation, feasible, risk=risk)
print(spi_decision.action.name)
```

For data-derived priors:

- `CountPriorBuilder` and `build_empirical_prior()` consume `(abstract_state, action)` samples.
- `build_model_prior()` consumes `(observation, longitudinal_acceleration, candidates, feasible)` samples and selects actions using MOBIL lane-change criteria. IDM helpers can supply longitudinal accelerations.
- `HybridPrior(model, empirical, weight=0.35)` combines model and empirical probabilities as `weight * model + (1 - weight) * empirical`. For weights strictly between 0 and 1, only states present in both tables are retained.

Prior integration has three input requirements:

1. **State coverage:** `FrozenPrior` raises `KeyError` for unseen states. Provide coverage or a callable with an explicit fallback.
2. **Positive probabilities for learning:** KL regularization requires every prior probability to be strictly positive. Count-based builders do not add smoothing; smooth zero counts before using their distributions in learning.
3. **Valid abstraction ranges:** quantization uses lower-inclusive, upper-exclusive intervals and raises `ValueError` outside them. Defaults are speed `[0, 33.3)`, leader gap `[0, 200)`, relative speed `[-10, 10)`, and merge distance `[0, 2000)`. Leader gap is neighbor distance minus vehicle length.

## Connecting a simulator

Create a local agent for each independently controlled vehicle. At each decision step:

1. Build its `LocalObservation`, feasible-action mask, and, when using a prior, action-specific risk vector.
2. Call `act()` and apply the resulting command through the simulator.
3. Compute reward and the next observation, then call `observe()`.
4. Update each agent after collecting its rollout or sufficient replay data.

For IPPO/IA2C, distinguish terminal states (`terminated=True`) from time-limit cutoffs (`truncated=True`). The advantage calculation bootstraps at truncations but stops advantage propagation across episode boundaries. Collect sampled actions for training.

The dynamics and reward modules provide individual calculations. The environment owns collision detection, lane-change steering, resets, episode boundaries, and multi-vehicle coordination.

For context-conditioned agents, pass a `ContextualInitializer` via `initializer=`, along with `initial_observation`, `role`, and `phase`. Default role and phase inputs have lengths 4 and 2; the role vector is separate from the six-slot neighbor `Role` enum. Agent factories copy and freeze the context encoder. Training the initializer and assigning meanings to these context inputs are tasks for the surrounding experiment.

## Configuration

Defaults live in [`marl_spi/config.py`](marl_spi/config.py):

| Configuration | Selected defaults |
| --- | --- |
| `PaperConfig` / `PAPER` | 21 observation features, 5 actions, 128 hidden units; discount 0.99, batch size 128, learning rate 0.0005 |
| `LearningConfig` | PPO clip 0.2, GAE λ 0.95, 10 update epochs; value coefficient 0.5, entropy coefficient 0.01 |
| `RuntimeConfig` / `RUNTIME` | Wheelbase 2.7 m, safe clearance 2 m, role input size 4, phase input size 2 |
| `DQNConfig` / `DQN` | Replay capacity 50,000, temperature 1.0, target update every 500 updates, epsilon 0.05 |

The agent factories' `config=` parameter accepts a `LearningConfig`. Other factory options include `device`, and IDQN replay capacity, temperature, and target-update interval. Shared `PAPER` and `RUNTIME` values are imported by modules and often captured as function defaults.

## Repository layout

```text
marl_spi/
├── actions.py          # Semantic actions and vehicle commands
├── observation.py      # Neighbor roles and local observation vectors
├── agent.py            # Actor-critic agent and factory
├── q_agent.py          # Q-learning agent and factory
├── config.py           # Shared and learner-specific defaults
├── dynamics.py         # Kinematics and clearance calculations
├── rewards.py          # Reward components and weighted total
├── learning/           # IPPO, IA2C, IDQN, buffers, GAE, behavior cloning
├── networks/           # Actor/critic, context encoder, hypernetwork, modulation
├── policies/           # Feasibility, soft masking, adaptive KL
└── priors/             # State abstraction, empirical/model/hybrid priors
```
