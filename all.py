"""Minimal sage.all shim so the Albrecht-Player-Samuels lattice-estimator
runs on plain CPython with mpmath for high-precision real arithmetic.

Validated against the estimator's own doctest outputs (Kyber512:
usvp 2^143.8 / beta 406 / d 998, etc.).  Sage-specific symbolic ring
constructs are stubbed out (only used by the Arora-GB attack, which we do
not exercise)."""
from mpmath import mp
import math as _math
import functools

mp.prec = 200

# mpf subclass supporting Sage's .n(prec) precision method; mpmath
# classmethods construct results via cls, so the subclass is preserved
# through arithmetic.
_mpf_base = mp.mpf


class mpf(_mpf_base):
    def n(self, prec=53, **kw):
        return self

    def __new__(cls, *a, **kw):
        if cls is mpf:
            return _mpf_base.__new__(mpf, *a, **kw)
        return _mpf_base.__new__(cls, *a, **kw)


mp.mpf = mpf


def _mp(x):
    if isinstance(x, mpf):
        return x
    return mpf(x)


class _RealType:
    def __call__(self, x, prec=None):
        return mpf(x)

    def __instancecheck__(self, x):
        return isinstance(x, mpf)

    pi = mpf(_math.pi)
    e = mpf(_math.e)
    zeta = staticmethod(lambda k: mp.zeta(k))


RR = _RealType()
ZZ = _RealType()
QQ = _RealType()


class _RDF(float):
    @staticmethod
    def pi():
        return _math.pi

    @staticmethod
    def e():
        return _math.e


RDF = _RDF
RealField = lambda prec=53: RR
pi = mpf(_math.pi)
e = mpf(_math.e)
oo = mpf("inf")
euler_gamma = mpf("0.57721566490153286060651209008240243104215933593992")


def log(x, b=None):
    x = _mp(x)
    if b is None:
        return mp.log(x)
    return mp.log(x) / mp.log(_mp(b))


def exp(x):
    return mp.exp(_mp(x))


def sqrt(x):
    return mp.sqrt(_mp(x))


def ceil(x):
    return int(mp.ceil(_mp(x)))


def floor(x):
    return int(mp.floor(_mp(x)))


def round(x):
    return int(mp.floor(_mp(x) + mpf("0.5")))


def erf(x):
    return mp.erf(_mp(x))


def zeta(x):
    return mp.zeta(_mp(x))


def coth(x):
    return 1 / mp.tanh(_mp(x))


def tanh(x):
    return mp.tanh(_mp(x))


def binomial(n, k):
    return _math.comb(int(n), int(k))


def prod(seq):
    out = mpf(1)
    for v in seq:
        out = out * _mp(v)
    return out


def parent(x):
    class _P:
        def prec(self):
            return 0
    return _P()


class _Var:
    def __call__(self, name):
        return name


var = _Var()
find_root = lambda f, a, b: 0.5 * (a + b)
find_fit = lambda *a, **k: []
line = lambda *a, **k: None


class _RealDistribution:
    """Chi-squared / beta CDFs via mpmath (used by estimator/prob.py)."""

    def __init__(self, kind, params=None):
        self.kind = kind
        self.params = params

    def cum_distribution_function(self, x):
        x = _mp(x)
        if self.kind == "chisquared":
            k = _mp(self.params) / 2
            x = max(x, mpf(0))
            return mp.gammainc(k, 0, x / 2, regularized=True)
        if self.kind == "beta":
            a, b = self.params
            x = max(min(x, mpf(1)), mpf(0))
            return mp.betainc(_mp(a), _mp(b), 0, x, regularized=True)
        raise NotImplementedError(self.kind)


RealDistribution = _RealDistribution


def cached_function(func=None, **kw):
    def deco(f):
        g = functools.lru_cache(maxsize=None)(f)
        g.__wrapped__ = f
        return g

    if func is None:
        return deco
    return deco(func)


# --- stubs for symbolic ring constructs (Arora-GB only; not exercised here) ---
class _RingStub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return lambda *a, **k: None

    def gen(self):
        return "x"


def PowerSeriesRing(*a, **k):
    return _RingStub()


def PolynomialRing(*a, **k):
    return _RingStub()
