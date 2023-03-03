# ecosys
[![ecosys](https://github.com/fcelli/ecosys/actions/workflows/python-app.yml/badge.svg)](https://github.com/fcelli/ecosys/actions/workflows/python-app.yml)

## Ecosys-v0 Environment

### Description

This environment describes the movement of a herbivore scavenging for resources on a square grid world.

### Action Space

The action is a `ndarray` with shape `(1,)` which can take values `{0, 1, 2, 3}` indicating the movement of the herbivore on the grid.

| Num | Action               |
|-----|----------------------|
| 0   | Move herbivore up    |
| 1   | Move herbivore right |
| 2   | Move herbivore down  |
| 3   | Move herbivore left  |

### Observation Space

The observation is a `ndarray` with shape `(2, 4)` with multibinary values `{0, 1}`. The first array describes the position of resources (food) with respect to the herbivore (importance of food scales with 1/distance<sup>2</sup>). The second array describes the position of the herbivore with respect to the grid boundary (wall).

| Idx  | Observation | Values |
|------|-------------|--------|
| 0, 0 | Food Up     | {0, 1} |
| 0, 1 | Food Right  | {0, 1} |
| 0, 2 | Food Down   | {0, 1} |
| 0, 3 | Food Left   | {0, 1} |
| 1, 0 | Wall Up     | {0, 1} |
| 1, 1 | Wall Right  | {0, 1} |
| 1, 2 | Wall Down   | {0, 1} |
| 1, 3 | Wall Left   | {0, 1} |

### Rewards

| Reward                              | Description                             |
|-------------------------------------|-----------------------------------------|
| $+100$                              | The herbivore eats all resources        |
| $+10$                               | The herbivore eats one resource         |
| $-100$                              | The herbivore crosses the grid boundary |
| $-\frac{1}{2(\text{grid-dim} - 1)}$ | Otherwise                               |

### Starting State

The herbivore and all resources are randomly scattered on the grid.

### Episode End

The episode ends if any one of the following occurs:
1. Termination: The herbivore crosses the grid boundary
2. Termination: The herbivore eats all resources on the grid
3. Truncation: Episode length is greater than 500

### Arguments

```
gym.make('Ecosys-v0')
```

## How to Install
```
git clone git@github.com:fcelli/ecosys.git
cd ecosys
make init
```

## Running Tests
```
make test
```

## Training a Model
```
make train
```

## Running the Simulation
```
make run
```

<img src="https://github.com/fcelli/ecosys/blob/main/docs/example.gif" width="40%" height="40%"/>