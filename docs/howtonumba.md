The challenge with `njit` is that it can't work with Python classes and `self`. The standard approach is to extract the core computation into a **standalone function** outside the class, compile that with `njit`, and then call it from your class methods.

Here's exactly what to do:

**Step 1: Write a standalone njit function above your class**

```python
from numba import njit

@njit
def _run_sweeps(spins, length_xy, beta, xs, ys, deltas, accepts, n_sweeps):
    """Compiled core loop: runs all sweeps, modifies spins in place."""
    steps_per_sweep = length_xy * length_xy
    for sweep in range(n_sweeps):
        for i in range(sweep * steps_per_sweep, (sweep + 1) * steps_per_sweep):
            x = xs[i]
            y = ys[i]
            initial_theta = spins[x, y]
            new_theta = deltas[i]

            xm = (x - 1) % length_xy
            xp = (x + 1) % length_xy
            ym = (y - 1) % length_xy
            yp = (y + 1) % length_xy

            energy_diff = (
                -np.cos(new_theta - spins[xm, y]) + np.cos(initial_theta - spins[xm, y])
                - np.cos(new_theta - spins[xp, y]) + np.cos(initial_theta - spins[xp, y])
                - np.cos(new_theta - spins[x, ym]) + np.cos(initial_theta - spins[x, ym])
                - np.cos(new_theta - spins[x, yp]) + np.cos(initial_theta - spins[x, yp])
            )

            if accepts[i] < np.exp(min(0.0, -beta * energy_diff)):
                spins[x, y] = new_theta
```

Note that `_sweep` and `_step` are fully absorbed into this one function — numba works best with flat, fused loops rather than nested function calls.

**Step 2: Modify `run` to call this function, sampling between intervals**

```python
def run(self, sweeps: int = 1000, store: bool = True, interval: int = 10):
    size = sweeps * self.length_xy**2
    rng = np.random.default_rng()
    xs = rng.integers(0, self.length_xy, size=size, dtype=np.int64)
    ys = rng.integers(0, self.length_xy, size=size, dtype=np.int64)
    deltas = rng.uniform(0, 2 * np.pi, size=size)
    accepts = rng.random(size=size)

    for i in tqdm(range(0, sweeps, interval)):
        if store:
            self.magn_hist.append(self._calculate_magnetization())
            self.spins_hist.append(self.spins.copy())  # note: copy() important here
        
        batch = min(interval, sweeps - i)  # handle case where sweeps % interval != 0
        _run_sweeps(self.spins, self.length_xy, self.beta,
                    xs, ys, deltas, accepts, 
                    n_sweeps=batch)  # won't work as is, see note below
```

Actually, since the random arrays are indexed by absolute sweep count, you need to also pass the offset. Simplest fix: **slice the arrays per batch**:

```python
    for i in tqdm(range(0, sweeps, interval)):
        if store:
            self.magn_hist.append(self._calculate_magnetization())
            self.spins_hist.append(self.spins.copy())
        
        batch = min(interval, sweeps - i)
        start = i * self.length_xy**2
        end = (i + batch) * self.length_xy**2
        _run_sweeps(self.spins, self.length_xy, self.beta,
                    xs[start:end], ys[start:end], deltas[start:end], accepts[start:end],
                    n_sweeps=batch)
```

And update `_run_sweeps` to index from 0 since sliced arrays always start at 0.

**Step 3: Warm up the JIT**

The first call compiles and will be slow (~seconds). You can trigger this explicitly at startup with a tiny dummy run:

```python
# At the bottom of your file, outside the class:
def warmup():
    dummy_spins = np.zeros((4, 4))
    _run_sweeps(dummy_spins, 4, 1.0,
                np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64),
                np.zeros(16), np.zeros(16), n_sweeps=1)

warmup()
```

This way your actual simulation runs don't pay the compilation cost.

**One important note on `spins_hist`:** make sure you append `self.spins.copy()` not `self.spins` — numba modifies the array in place, so without `.copy()` all history entries will point to the same array and reflect the final state.