# ORBIT

## Setup `uv`

```sh
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
bash <(curl -s https://astral.sh/uv/install.sh)
```

Verify installation:
```sh
uv --version
```

## Install ORBIT
Optionally set up virtual environment, and then activate
```sh
uv venv
```
Using the .whl file provided by Quantum Dice
```sh
uv pip install path/to/file.whl
```

## Basic Usage
```python
import orbit
import numpy as np

# Interaction Matrix
J = np.array(
    [
        [ 0,  1, -1,  0],
        [ 1,  0,  0,  1],
        [-1,  0,  0, -1],
        [ 0,  1, -1,  0]
    ]
)

# External field
h = np.zeros(4)

result = orbit.optimize_ising(J,
                              h,
                              n_replicas=1,
                              full_sweeps=100,
                              beta_initial=0.5,
                              beta_end=2,
                              beta_step_interval=1)

print(f'Minimum energy found: {result.min_cost}')
print(f'Minimum energy state: {result.min_state}')
```
