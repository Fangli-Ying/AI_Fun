# Rocket Car Landing Lab

A small, self-contained **Q-learning demonstration** built with Python and Pygame. A rocket-powered car must learn to land on a pad while controlling its position, speed, tilt, and rotation.

The project is designed for live demonstrations and classroom discussion. Students can watch the agent learn from rewards and crashes, inspect its changing exploration rate and Q-table size, or fly the car manually.

## Requirements

- Python 3.9 or later is recommended.
- Pygame.

Install the dependency:

```bash
python -m pip install pygame
```

## Run the simulation

Place `rocket_car_landing.py` and (optionally) a saved model file in the same directory, then run:

```bash
python rocket_car_landing.py
```

The graphical window opens with a start screen. Press **Enter** to train the agent. By default, the GUI training target is 5,000 episodes.

### Command-line options

| Option | What it does | Example |
|---|---|---|
| `--episodes N` | Sets the training target used by the GUI or the number of headless training episodes. | `python rocket_car_landing.py --episodes 2000` |
| `--seed N` | Sets the random seed for the agent and environment. Reusing a seed helps make runs more comparable, but does not guarantee identical results if other settings or saved models differ. | `python rocket_car_landing.py --seed 12` |
| `--model PATH` | Chooses where to load/save the Q-table. The default is `lander_q_table.pkl` beside the script. | `python rocket_car_landing.py --model demo.pkl` |
| `--fresh` | Ignores an existing saved Q-table and starts with a new agent. Useful for showing learning from scratch. | `python rocket_car_landing.py --fresh` |
| `--headless` | Trains without opening the graphical window. | `python rocket_car_landing.py --headless --episodes 5000` |
| `--evaluate N` | After headless training, runs `N` greedy evaluation flights and prints the safe-landing results. Use with `--headless`. | `python rocket_car_landing.py --headless --episodes 5000 --evaluate 100` |

A useful preparation run is:

```bash
python rocket_car_landing.py --headless --episodes 5000 --evaluate 100 --model class_demo.pkl
```

Then show the saved policy in the GUI:

```bash
python rocket_car_landing.py --model class_demo.pkl
```

**Important:** If you want the audience to see learning from scratch, use `--fresh` or choose a new model filename. Otherwise, the program loads an existing Q-table when one is present. During GUI training, the program saves the learned model automatically when it reaches its training target; press **S** to save manually.

## Keyboard controls

### Main modes

| Key | Action |
|---|---|
| `Enter` | Start/restart training |
| `D` | Watch the agent choose actions without random exploration |
| `M` | Switch to manual control |
| `F` | Toggle fast/normal training speed |
| `P` | Pause/resume |
| `R` | Restart the current mode/flight |
| `S` | Save the current Q-table |
| `Esc` | Quit |

Fast training processes many simulation steps per rendered frame, so it is useful when the class needs to see learning progress quickly. Normal speed is better for observing individual actions.

### Manual flight

| Key | Control |
|---|---|
| `Space` | Main thruster |
| `A` / `D` | Left/right side thruster |
| `Down Arrow` | Top thruster |

In manual mode, the car responds to the keys rather than learning from them. This is useful for demonstrating how thrust, gravity, and rotation affect the landing.

## What the agent learns

The environment measures six continuous quantities and maps them into discrete bins to form a Q-table state:

- Horizontal displacement from the pad
- Altitude
- Horizontal velocity
- Vertical velocity
- Tilt angle
- Angular velocity (spin)

The agent chooses among seven actions: coast, main thruster, left or right side thruster, main plus left/right, and top thruster. It uses **epsilon-greedy Q-learning**: with probability `epsilon` it explores a random action; otherwise it chooses an action with the highest current Q-value.

The reward combines improvement in a shaped flight-cost estimate, a small time penalty, a small main-thruster fuel penalty, and a large final reward or penalty for the outcome. The flight-cost estimate supplies reward feedback; it does **not** directly select actions.

A safe landing requires all of the following: touching down within the pad, low horizontal and vertical speed, a sufficiently small tilt, and low angular velocity. Merely reaching the ground is not enough.

## Parameters to change for classroom demonstrations

The most convenient parameters are grouped near the top of `rocket_car_landing.py`. Change one group at a time, save the file, and rerun the program. Results can vary because Q-learning is stochastic and training may not converge within the chosen episode count.

### 1. Training and learning

These are in `QLearningAgent.__init__`:

| Parameter | Default | Meaning | Demonstration idea |
|---|---:|---|---|
| `alpha` | `0.22` | Learning rate: how strongly a new update changes a Q-value. It gradually decreases after each episode, with a floor of `0.03`. | Compare faster initial updates with a smaller starting value. |
| `gamma` | `1.0` | Discount factor for future rewards. | Discuss how immediate versus future outcomes are weighted. |
| `epsilon` | `1.0` | Initial probability of taking a random exploratory action. | Start lower to reduce early randomness, or keep at `1.0` to show exploration. |
| `epsilon_min` | `0.05` | Minimum exploration probability. | Raise it to keep more exploration, or lower it to make the policy more consistently greedy later. |
| `epsilon_decay` | `0.995` | Multiplier applied to epsilon after each training episode. | A smaller value reduces exploration more quickly; a value closer to `1` preserves it longer. |

The learning rate also decays in `finish_episode()`:

```python
self.alpha = max(0.03, self.alpha * 0.9995)
```

The `0.9995` multiplier controls how quickly the learning rate decreases; `0.03` is its lower bound. The headless and GUI training paths both call the episode-finish update.

**Classroom comparison:** Run two trials with the same episode count and seed, but change only `epsilon_decay`. Compare the displayed recent landing rate and, if using headless evaluation, the number of safe landings. For a fair comparison, start each trial with a fresh model.

### 2. Physics

These constants are near the top of the file:

| Parameter | Default | Effect |
|---|---:|---|
| `GRAVITY` | `9.8` | Downward acceleration. Increasing it makes the descent harder to brake. |
| `MAIN_ACCEL` | `18.0` | Main-thruster acceleration. |
| `SIDE_ACCEL` | `3.5` | Horizontal acceleration from side thrusters. |
| `TOP_ACCEL` | `7.0` | Downward acceleration from the top thruster. |
| `SIDE_TORQUE` | `0.65` | Rotational acceleration caused by side-thruster actions. |
| `DT` | `0.05` | Duration of one physics step, in seconds. |
| `MAX_STEPS` | `800` | Maximum number of steps per flight (40 seconds at the default `DT`). |

**Classroom comparisons:**
- Increase `GRAVITY` while keeping the thrusters unchanged.
- Reduce `MAIN_ACCEL` to make braking harder.
- Increase `SIDE_ACCEL` to strengthen horizontal correction.
- Increase `SIDE_TORQUE` to make side-thruster actions change rotation more strongly.

Changing physics changes the task the agent must learn. A Q-table trained under the old physics may perform poorly under the new settings, so use `--fresh` or a new `--model` file.

Changing `DT` also changes how much simulated time each step represents and affects the numerical behavior of the physics. It is a less straightforward first experiment than changing a single acceleration constant.

### 3. Safe-landing criteria

These constants define the landing envelope:

| Parameter | Default | Meaning |
|---|---:|---|
| `PAD_HALF_WIDTH` | `17.0` | Half-width of the landing pad, in world units. |
| `SAFE_VX` | `2.0` | Maximum permitted absolute horizontal speed at touchdown. |
| `SAFE_VY` | `2.2` | Maximum permitted absolute vertical speed at touchdown. |
| `SAFE_ANGLE` | `0.16` radians | Maximum permitted absolute tilt (about 9 degrees). |
| `SAFE_OMEGA` | `0.8` | Maximum permitted absolute angular velocity. |

**Classroom comparison:** Tighten one limit, such as `SAFE_VY`, and discuss why a landing that previously counted as successful may now count as a crash. The landing test is separate from the agent's learned policy: changing the criteria changes how success is judged, even if the flight itself is unchanged.

### 4. State discretization

These bin boundaries are defined in `RocketCarEnv`:

```python
DX_BINS = (-40, -10, 10, 40)
ALTITUDE_BINS = (0, 10, 30, 70, 140)
VX_BINS = (-4, -1, 1, 4)
VY_BINS = (-10, -3, -1, 1, 5)
ANGLE_BINS = (-0.25, -0.08, 0.08, 0.25)
OMEGA_BINS = (-0.5, -0.1, 0.1, 0.5)
```

Each tuple divides one measurement into discrete ranges. More boundaries create more possible states and can represent situations in greater detail, but also make the Q-table larger and can leave the agent with less experience per state. Fewer boundaries reduce the number of states but combine more distinct situations into the same state.

**Classroom comparison:** Add a boundary to `VX_BINS` or remove one, then observe how the Q-table size and learning behavior change.

**Important:** Start with a fresh model after changing the bins. A saved Q-table was learned using the old state encoding, and the program does not automatically detect a mismatch in bin definitions.

### 5. Reward design

The reward is calculated in `RocketCarEnv.step()` and `flight_cost()`:

| Code/value | Meaning |
|---|---|
| `8.0 * (previous_cost - new_cost)` | Rewards improvement in the estimated flight cost. |
| `-0.05` | Small per-step cost discouraging endless hovering. |
| `-0.003` when the main thruster is active | Small fuel-use penalty. |
| `+300.0` | Reward for a safe landing. |
| `-80.0 - 100.0 * impact_cost` | Penalty for a failed touchdown or leaving the valid flight area. |
| `-500.0` | Additional penalty for reaching `MAX_STEPS` without ending the flight. |

The weights inside `flight_cost()` determine how much the shaping signal values position, altitude, speed, tilt, and spin. For example, `position_cost` is weighted by `1.5`, while `attitude_cost` is weighted by `0.8`.

**Classroom comparison:** Change one reward weight or the main-thruster penalty and ask students to predict the effect before running training. Reward changes can have surprising effects, so compare outcomes rather than assuming a particular change must improve performance.

### 6. Starting difficulty and episode count

The `difficulty` value is calculated during training and passed to `env.reset(difficulty)`. It increases from easier starting conditions to the full range of starting conditions over roughly the first 60% of the target episode count. The landing test and available actions do not change with this curriculum; the starting state does.

Useful settings:
- `--episodes 1000` for a shorter trial.
- `--episodes 5000` for the default-sized run.
- `--episodes 10000` for a longer run, if class time permits.

More episodes do not guarantee a better policy in every run. The program keeps a checkpoint based on recent landing success and restores the best recorded checkpoint at the end of training.

## Suggested classroom demonstrations

1. **Explore vs. exploit:** Explain `epsilon`, then compare a fresh run with a high exploration rate to one where exploration decays faster.
2. **Easy vs. difficult physics:** Increase gravity or reduce main-thruster acceleration, retrain, and compare the landing rate.
3. **What counts as success?** Tighten `SAFE_VY` or `SAFE_ANGLE` and show that the same touchdown can be judged differently.
4. **State-space size:** Change one bin tuple and observe the Q-state count and training behavior.
5. **Reward shaping:** Change one reward weight, predict the result, and test the prediction.
6. **Manual vs. learned control:** Use `M` to fly manually, then `D` to watch the learned greedy policy.

For comparisons, change **one parameter at a time**, use the same seed where appropriate, start with a fresh Q-table, and compare multiple runs or evaluate a fixed number of test flights. The recent landing rate shown in the GUI is a rolling window, not a guarantee of future performance.

## Files and saved models

- `rocket_car_landing.py` — simulation, renderer, and Q-learning implementation.
- `lander_q_table.pkl` — default saved Q-table created by training or the save command.

The saved model is a Python pickle file. Only load model files you trust. If the available action names do not match the current code, the program rejects the model; changes to other settings, such as physics or state bins, may not be detected automatically.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'pygame'`:** Install Pygame with `python -m pip install pygame`.
- **The demo starts with an already-trained agent:** Use `--fresh` or select a new `--model` path.
- **The agent crashes frequently after a code change:** Start fresh and allow it to retrain. Check whether the change made the task substantially harder.
- **Headless training does not open a window:** This is expected when `--headless` is supplied.
- **Training is too fast to explain:** Press `F` to switch out of fast training.

## Educational note

This is a compact demonstration of tabular Q-learning, not a high-fidelity rocket simulator. It discretizes continuous measurements, uses a hand-designed reward signal, and learns through repeated trials. These simplifications make the agent's state, actions, rewards, and learning process easier to discuss in class.
