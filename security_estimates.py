"""Run the Albrecht-Player-Samuels lattice estimator (with sage shim) on
DUOLITH-KEM's wall parameters and on published anchor schemes, to position
the new construction against the literature.

Anchors (published parameters):
  ML-KEM (FIPS 203): module-LWE, n = 256k, q = 3329.  Key-generation noise is
    CBD(eta1) for both s and e: eta1 = 3 for ML-KEM-512, eta1 = 2 for
    ML-KEM-768/1024.
  FrodoKEM-640:  plain LWE, n = 640, q = 2^15, sigma = 2.8
  FrodoKEM-976:  plain LWE, n = 976, q = 2^16, sigma = 2.3
  (Frodo uses a CDF-based discrete distribution; we approximate with a
   DiscreteGaussian of the same standard deviation.)

DUOLITH-KEM walls (current parameter sets):
  Wall-S: module-LWE over Z_7681[X]/(X^256+1), rank k in {2,3}, CBD(3)
  Wall-U: plain LWE over Z_{2^16}, n in {640,768}, CBD(8) (sigma = 2)

Setup:
  git clone https://github.com/malb/lattice-estimator
  LATTICE_ESTIMATOR=/path/to/lattice-estimator python3 security_estimates.py

If a real Sage install is not importable, the bundled shim (all.py, next to
this file) is registered as `sage.all` automatically.
"""
import importlib.util
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("LATTICE_ESTIMATOR",
                                  os.path.join(_HERE, "lattice-estimator")))


def _install_sage_shim():
    try:
        import sage.all  # noqa: F401
        return
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        "sage.all", os.path.join(_HERE, "all.py"))
    shim = importlib.util.module_from_spec(spec)
    pkg = types.ModuleType("sage")
    pkg.__path__ = []
    pkg.all = shim
    sys.modules["sage"] = pkg
    sys.modules["sage.all"] = shim
    spec.loader.exec_module(shim)


_install_sage_shim()

from estimator import LWE, ND  # noqa: E402
from estimator.lwe_parameters import LWEParameters  # noqa: E402


def estimate(name, n, q, sigma_s, sigma_e, m=None):
    p = LWEParameters(n=n, q=q,
                      Xs=ND.DiscreteGaussian(sigma_s),
                      Xe=ND.DiscreteGaussian(sigma_e),
                      m=m if m is not None else n)
    print("\n== %s: n=%d q=%d sigma_s=%s sigma_e=%s" % (name, n, q, sigma_s, sigma_e))
    try:
        r = LWE.estimate(p)
        costs = {}
        for tag, c in r.items():
            costs[tag] = float(np.log2(float(c["rop"])))
        best = min(v for v in costs.values() if v is not None)
        for tag, v in sorted(costs.items(), key=lambda kv: kv[1] if kv[1] else 9e9):
            print("   %-14s 2^%.1f" % (tag, v) if v else "   %-14s n/a" % tag)
        print("   BEST: 2^%.1f" % best)
        return best
    except Exception as exc:
        print("   estimate failed: %r" % exc)
        return None


def estimate_exact(name, n, q, cbd_eta_s, cbd_eta_e, m=None):
    """CBD noise: sigma = sqrt(eta/2)."""
    p = LWEParameters(n=n, q=q,
                      Xs=ND.CenteredBinomial(cbd_eta_s),
                      Xe=ND.CenteredBinomial(cbd_eta_e),
                      m=m if m is not None else n)
    print("\n== %s: n=%d q=%d CBD(%d)/CBD(%d) sigma=%.3f/%.3f"
          % (name, n, q, cbd_eta_s, cbd_eta_e,
             (cbd_eta_s / 2) ** 0.5, (cbd_eta_e / 2) ** 0.5))
    try:
        r = LWE.estimate(p)
        costs = {}
        for tag, c in r.items():
            costs[tag] = float(c["rop"].log(2))
        best = min(costs.values())
        for tag, v in sorted(costs.items(), key=lambda kv: kv[1]):
            print("   %-14s 2^%.1f" % (tag, v))
        print("   BEST: 2^%.1f" % best)
        return best
    except Exception as exc:
        print("   estimate failed: %r" % exc)
        return None


if __name__ == "__main__":
    print("#" * 76)
    print("# ANCHORS (published parameter sets)")
    print("#" * 76)
    # ML-KEM: q=3329; keygen noise CBD(eta1): eta1=3 (512), eta1=2 (768/1024)
    estimate_exact("ML-KEM-512 (anchor)", 512, 3329, 3, 3)
    estimate_exact("ML-KEM-768 (anchor)", 768, 3329, 2, 2)
    estimate_exact("ML-KEM-1024 (anchor)", 1024, 3329, 2, 2)
    # FrodoKEM (approximated as discrete Gaussian of the same sigma):
    #   640: q=2^15, sigma=2.8;  976: q=2^16, sigma=2.3
    estimate("FrodoKEM-640 (anchor, approx)", 640, 2**15, 2.8, 2.8)
    estimate("FrodoKEM-976 (anchor, approx)", 976, 2**16, 2.3, 2.3)

    print()
    print("#" * 76)
    print("# DUOLITH-KEM walls (current parameters)")
    print("#" * 76)
    estimate_exact("Wall-S / DUOLITH-128", 512, 7681, 3, 3)
    estimate_exact("Wall-S / DUOLITH-192", 768, 7681, 3, 3)
    estimate_exact("Wall-U / DUOLITH-128", 640, 2**16, 8, 8)
    estimate_exact("Wall-U / DUOLITH-192", 768, 2**16, 8, 8)
