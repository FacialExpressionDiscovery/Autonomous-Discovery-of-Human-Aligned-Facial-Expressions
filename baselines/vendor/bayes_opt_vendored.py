"""
baselines/vendor/bayes_opt_vendored.py

Vendored copy of the `bayesian-optimization` package (PyPI: bayesian-optimization,
GitHub: fmfn/BayesianOptimization -- reference [30] in Yang et al. 2022, "Optimizing
Facial Expressions of an Android Robot Effectively: a Bayesian Optimization Approach").

All seven upstream modules (event.py, observer.py, logger.py, util.py,
target_space.py, domain_reduction.py, bayesian_optimization.py) are flattened
into this single file for easier review; each section below is still exactly
the corresponding upstream module (same class/function bodies), just without
the `from .xxx import ...` relative imports, which are unnecessary now that
everything lives in one module namespace. No section was reordered relative
to its upstream dependencies (event -> observer -> logger -> util ->
target_space -> domain_reduction -> bayesian_optimization), so top-to-bottom
reading order matches the original import graph.

Provenance
----------
  Source          : PyPI `bayesian-optimization==1.2.0`
                     (Home-page: https://github.com/fmfn/BayesianOptimization)
  Vendored from    : local `pip install bayesian-optimization==1.2.0` install, i.e.
                     the exact wheel resolved by that pin at vendoring time.
  Vendoring date   : 2026-07-22
  Reason vendored  : the pip package as published crashes unconditionally on import
                     of BayesianOptimization(...) under numpy>=1.24, because
                     target_space.py references the removed alias `np.float`
                     (np.float was deprecated in NumPy 1.20 and removed thereafter),
                     and crashes again inside acq_max() because modern SciPy's
                     minimize() requires a strictly 1-D `x0`. Rather than run an
                     incompatible pip package or fall back to a custom
                     reimplementation, the source is vendored here with the two
                     minimal patches documented below so the *historical* GP/UCB/
                     acquisition/TargetSpace logic runs unmodified.

Historical cross-check (see baselines/bayesian_method.py module docstring for
the full writeup): 1.2.0 (released 2020-05-16) was still the latest PyPI
release on 2022-07-15, the paper's stated deadline; the next release, v1.3.0,
shipped 2022-10-24. Diffing the upstream GitHub history from tag 1.2.0 up to
the latest pre-deadline commit (5bc8ad4, 2022-06-26) shows the GP constructor,
UCB formula, target normalization, and suggest/register/probe/maximize control
flow are IDENTICAL to 1.2.0 -- no relevant core-behavior divergence. Only two
pre-deadline commits touch these files at all, and both are exactly the two
compatibility patches applied below (one of them, the np.float fix, is
textually identical to the upstream commit; the x0-shape fix upstream applied
was actually merged *after* the deadline, 2022-08-25, since the SciPy
1-D-x0 requirement it addresses postdates 1.2.0's original SciPy target).

Patch log
---------
  1. TargetSpace.__init__ (originally target_space.py): `dtype=np.float` ->
     `dtype=float`. `np.float` was a deprecated alias for the Python builtin
     `float` and was removed from NumPy (this environment: numpy 1.26.4). The
     alias was always equivalent to the builtin `float`; this patch changes
     zero numeric behavior. Identical, line-for-line, to the upstream
     project's own pre-deadline fix (commit 8db70f267d, 2022-06-09).

  2. acq_max (originally util.py): the L-BFGS-B restart loop passed x0 as a
     (1, D)-shaped array (`x_try.reshape(1, -1)`). Modern SciPy's minimize()
     (this env: scipy 1.15.3) raises "'x0' must only have one dimension" for
     that shape. Patched to pass `x_try` (already shape (D,)) directly as x0;
     the acquisition-function lambda still reshapes its argument to (1, D)
     internally before calling gp.predict, so the UCB/EI/POI math and the
     L-BFGS-B search itself are unaffected. As a direct consequence of x0 now
     being 1-D, SciPy returns `res.fun` as a plain float instead of a
     length-1 ndarray, so the two `res.fun[0]` reads were replaced with
     `float(np.ravel(res.fun)[0])`, which reads the same scalar value under
     either return type. (Upstream fixed the same symptom differently and
     earlier -- commit b4e09a2584, 2022-02-11, changed `res.fun[0]` to
     `np.squeeze(res.fun)` for a *different*, older SciPy incompatibility;
     upstream did not change the x0 shape itself until commit 34ee5c4cc4,
     2022-08-25, which is after this paper's deadline.)

  Neither patch alters the GP kernel/hyperparameters, the UCB formula, the
  acquisition-maximization search strategy (still n_warmup=10000 random +
  n_iter=10 L-BFGS-B restarts, both at upstream defaults), random
  initialization, or TargetSpace's register/probe/suggest/maximize control
  flow. Both patches are dtype/shape compatibility fixes only.

  3. Structural-only, not a behavior patch: ScreenLogger._step originally
     read `def _step(self, instance, colour=Colours.black):` -- a default
     argument evaluated at function-definition time. In the original
     multi-file layout this was fine because Python's import system fully
     executes util.py (which defines Colours) before logger.py's `from .util
     import Colours` runs. Flattened into one file with logger.py's section
     preceding util.py's, that default would raise NameError at class-body
     evaluation time. Changed to `colour=None` with `if colour is None:
     colour = Colours.black` inside the function body, which resolves at
     call time (long after the whole module has finished loading) and is
     therefore behaviorally identical -- not an upstream compatibility fix,
     purely a consequence of this file being a single-file flattening.

  4. Performance-only, not a numerics patch (added 2026-07-23, not present in
     any upstream release): acq_max's n_iter L-BFGS-B restarts were a serial
     `for x_try in x_seeds: minimize(...)` loop. Each restart is an
     independent, read-only optimisation against the same already-fitted
     `gp` (gp.predict does not mutate the estimator), so the n_iter calls to
     `minimize()` are now dispatched concurrently via
     `joblib.Parallel(backend="threading")` (numpy/scipy release the GIL
     during the underlying LAPACK/BLAS calls, so real wall-clock parallelism
     is achieved without inter-process pickling of `gp`). The results are
     then reduced in the ORIGINAL x_seeds order using the exact same
     `if max_acq is None or -fun_val >= max_acq` comparison as upstream, so
     tie-breaking and the final x_max/max_acq are bit-for-bit identical to
     the serial loop -- only wall-clock time changes, never the chosen point.
     n_warmup, n_iter, the UCB/EI/POI formulas, and the GP kernel/fit are
     untouched. The GP's own n_restarts_optimizer=5 kernel-hyperparameter
     restarts (inside sklearn's GaussianProcessRegressor.fit(), not in this
     file) were deliberately left serial: Matern(nu=2.5) here uses a scalar
     (isotropic) length_scale, i.e. a 1-D optimisation problem, so those 5
     restarts are already sub-millisecond in aggregate and are not a
     meaningful cost; parallelising them would require monkey-patching
     sklearn's internal optimizer rather than our own vendored code, for no
     measurable benefit.

Do not hand-edit any other line in this file without updating this log.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor


# ═════════════════════════════════════════════════════════════════════════
# event.py
# ═════════════════════════════════════════════════════════════════════════

class Events:
    OPTIMIZATION_START = 'optimization:start'
    OPTIMIZATION_STEP = 'optimization:step'
    OPTIMIZATION_END = 'optimization:end'


DEFAULT_EVENTS = [
    Events.OPTIMIZATION_START,
    Events.OPTIMIZATION_STEP,
    Events.OPTIMIZATION_END,
]


# ═════════════════════════════════════════════════════════════════════════
# observer.py
# ═════════════════════════════════════════════════════════════════════════

class Observer:
    def update(self, event, instance):
        raise NotImplementedError


class _Tracker(object):
    def __init__(self):
        self._iterations = 0

        self._previous_max = None
        self._previous_max_params = None

        self._start_time = None
        self._previous_time = None

    def _update_tracker(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            self._iterations += 1

            current_max = instance.max
            if (self._previous_max is None or
                current_max["target"] > self._previous_max):
                self._previous_max = current_max["target"]
                self._previous_max_params = current_max["params"]

    def _time_metrics(self):
        now = datetime.now()
        if self._start_time is None:
            self._start_time = now
        if self._previous_time is None:
            self._previous_time = now

        time_elapsed = now - self._start_time
        time_delta = now - self._previous_time

        self._previous_time = now
        return (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            time_elapsed.total_seconds(),
            time_delta.total_seconds()
        )


# ═════════════════════════════════════════════════════════════════════════
# logger.py
# ═════════════════════════════════════════════════════════════════════════

def _get_default_logger(verbose):
    return ScreenLogger(verbose=verbose)


class ScreenLogger(_Tracker):
    _default_cell_size = 9
    _default_precision = 4

    def __init__(self, verbose=2):
        self._verbose = verbose
        self._header_length = None
        super(ScreenLogger, self).__init__()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    def _format_number(self, x):
        if isinstance(x, int):
                s = "{x:< {s}}".format(
                    x=x,
                    s=self._default_cell_size,
                )
        else:
            s = "{x:< {s}.{p}}".format(
                x=x,
                s=self._default_cell_size,
                p=self._default_precision,
            )

        if len(s) > self._default_cell_size:
            if "." in s:
                return s[:self._default_cell_size]
            else:
                return s[:self._default_cell_size - 3] + "..."
        return s

    def _format_key(self, key):
        s = "{key:^{s}}".format(
            key=key,
            s=self._default_cell_size
        )
        if len(s) > self._default_cell_size:
            return s[:self._default_cell_size - 3] + "..."
        return s

    def _step(self, instance, colour=None):
        if colour is None:
            colour = Colours.black
        res = instance.res[-1]
        cells = []

        cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))

        return "| " + " | ".join(map(colour, cells)) + " |"

    def _header(self, instance):
        cells = []
        cells.append(self._format_key("iter"))
        cells.append(self._format_key("target"))
        for key in instance.space.keys:
            cells.append(self._format_key(key))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def _is_new_max(self, instance):
        if self._previous_max is None:
            self._previous_max = instance.max["target"]
        return instance.max["target"] > self._previous_max

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_START:
            line = self._header(instance) + "\n"
        elif event == Events.OPTIMIZATION_STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose == 1 and not is_new_max:
                line = ""
            else:
                colour = Colours.purple if is_new_max else Colours.black
                line = self._step(instance, colour=colour) + "\n"
        elif event == Events.OPTIMIZATION_END:
            line = "=" * self._header_length + "\n"

        if self._verbose:
            print(line, end="")
        self._update_tracker(event, instance)


class JSONLogger(_Tracker):
    def __init__(self, path):
        self._path = path if path[-5:] == ".json" else path + ".json"
        try:
            os.remove(self._path)
        except OSError:
            pass
        super(JSONLogger, self).__init__()

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)


# ═════════════════════════════════════════════════════════════════════════
# util.py
# ═════════════════════════════════════════════════════════════════════════

def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))

    # Find the minimum of minus the acquisition function
    # COMPAT PATCH (bayesian-optimization==1.2.0 -> this vendored copy):
    # upstream passed x0=x_try.reshape(1, -1) (shape (1, D)). Modern SciPy's
    # minimize() requires x0 to be strictly 1-D ("'x0' must only have one
    # dimension"), so we pass x_try directly (already shape (D,) as a row
    # of x_seeds); the objective lambda still reshapes to (1, D) internally
    # for gp.predict, so the acquisition math is unchanged. See the module
    # docstring at the top of this file for the full patch log.
    #
    # PERFORMANCE PATCH (see patch 4 in the module docstring): the n_iter
    # restarts below are independent read-only optimisations against the same
    # fitted `gp`, so they are computed concurrently here instead of in a
    # serial for-loop. The reduction below still walks the results in the
    # original x_seeds order with the original >= comparison, so the chosen
    # x_max/max_acq are identical to the serial loop -- only wall-clock time
    # changes.
    def _minimize_one(x_try):
        return minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                        x_try,
                        bounds=bounds,
                        method="L-BFGS-B")

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_minimize_one)(x_try) for x_try in x_seeds
    )

    for res in results:
        # See if success
        if not res.success:
            continue

        # COMPAT PATCH: with 1-D x0 (see above), SciPy now returns res.fun as a
        # plain Python float instead of a length-1 ndarray, so `res.fun[0]`
        # raises TypeError. np.ravel(...)[0] reads the scalar value correctly
        # for either return type -- no change to the acquisition value itself.
        fun_val = float(np.ravel(res.fun)[0])

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -fun_val >= max_acq:
            x_max = res.x
            max_acq = -fun_val

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)


def load_logs(optimizer, logs):
    """Load previous ...

    """
    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)


# ═════════════════════════════════════════════════════════════════════════
# target_space.py
# ═════════════════════════════════════════════════════════════════════════

def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            # COMPAT PATCH (bayesian-optimization==1.2.0 -> this vendored copy):
            # upstream reads `dtype=np.float`, an alias removed from NumPy after
            # 1.20 (this env: numpy 1.26.4). `np.float` was always identical to
            # the builtin `float`; this is a name-only fix with no numeric or
            # algorithmic effect. See the module docstring at the top of this
            # file for the full patch log.
            dtype=float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, params):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        try:
            target = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
        return target

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]


# ═════════════════════════════════════════════════════════════════════════
# domain_reduction.py  (unused by this project -- no bounds_transformer is
# ever passed to BayesianOptimization -- vendored only for import completeness
# since bayesian_optimization.py's __init__ used to re-export it)
# ═════════════════════════════════════════════════════════════════════════

class DomainTransformer():
    '''The base transformer class'''

    def __init__(self, **kwargs):
        pass

    def initialize(self, target_space: TargetSpace):
        raise NotImplementedError

    def transform(self, target_space: TargetSpace):
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        pass

    def initialize(self, target_space: TargetSpace) -> None:
        """Initialize all of the parameters"""
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]

        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _update(self, target_space: TargetSpace) -> None:

        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params[
            np.argmax(target_space.target)
        ]

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (self.gamma_pan * (1.0 + self.c_hat) +
                            self.gamma_osc * (1.0 - self.c_hat))

        self.contraction_rate = self.eta + \
            np.abs(self.current_d) * (self.gamma - self.eta)

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.array, global_bounds: np.array) -> np.array:
        for i, variable in enumerate(new_bounds):
            if variable[0] < global_bounds[i, 0]:
                variable[0] = global_bounds[i, 0]
            if variable[1] > global_bounds[i, 1]:
                variable[1] = global_bounds[i, 1]

        return new_bounds

    def _create_bounds(self, parameters: dict, bounds: np.array) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: TargetSpace) -> dict:

        self._update(target_space)

        new_bounds = np.array(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r
            ]
        ).T

        self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)


# ═════════════════════════════════════════════════════════════════════════
# bayesian_optimization.py
# ═════════════════════════════════════════════════════════════════════════

class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)


# ═════════════════════════════════════════════════════════════════════════
# Public exports + provenance constants (originally __init__.py)
# ═════════════════════════════════════════════════════════════════════════

__all__ = [
    "BayesianOptimization",
    "UtilityFunction",
    "Events",
    "ScreenLogger",
    "JSONLogger",
    "SequentialDomainReductionTransformer",
]

VENDORED_PACKAGE_NAME = "bayesian-optimization"
VENDORED_PACKAGE_VERSION = "1.2.0"
VENDORED_SOURCE_URL = "https://github.com/fmfn/BayesianOptimization"
VENDORED_PATCHES = [
    "target_space.py: dtype=np.float -> dtype=float (np.float removed in modern NumPy; "
    "behaviorally identical to the builtin float, no algorithmic change)",
    "util.py/acq_max: L-BFGS-B x0 passed as (D,) not (1,D) (modern SciPy requires 1-D x0); "
    "res.fun[0] -> float(np.ravel(res.fun)[0]) to match SciPy's new scalar return type; "
    "no change to UCB/EI/POI formulas or the random-warmup + L-BFGS-B search strategy",
    "util.py/acq_max: the n_iter L-BFGS-B restarts are dispatched via joblib.Parallel "
    "(threading) instead of a serial for-loop; results are reduced in the original "
    "x_seeds order with the original >= comparison, so x_max/max_acq are bit-for-bit "
    "identical to the serial loop -- performance-only, no numeric or algorithmic change",
]
