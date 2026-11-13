"""Run the Albrecht-Player-Samuels lattice estimator (with sage shim) on
DUOLITH-KEM's wall parameters and on published anchor schemes, to position
the new construction against the literature.

Anchors (published parameters):
  ML-KEM-512/768 (FIPS 203): module-LWE, n = 256k, q = 3329, CBD(2), sigma=1.0
  FrodoKEM-640/976: plain LWE, q = 2^15, discrete-Gaussian-style noise
  (Frodo uses a CDF-based discrete distribution; we approximate with
   DiscreteGaussian of the same standard deviation).

DUOLITH-KEM walls (current parameter sets):
  Wall-S: module-LWE over Z_7681[X]/(X^256+1), rank k in {2,3}, CBD(3)
  Wall-U: plain LWE over Z_{2^16}, n in {640,768}, CBD(8) (sigma = 2)
"""
import sys

sys.path.insert(0, "/tmp/le")
sys.path.insert(0, "/home/z/my-project/scripts/sage_shim")

from estimator import LWE, ND  # noqa: E402
from estimator.lwe_parameters import LWEParameters  # noqa: E402


def estimate(name, n, q, sigma_s, sigma_e, m=None):
    p = LWEParameters(n=n, q=q,
                      Xs=ND.DiscreteGaussian(sigma_s) if sigma_s % 1 else ND.CenteredBinomial(int(2 * sigma_s * 2)),
                      Xe=ND.DiscreteGaussian(sigma_e),
                      m=m if m is not None else n)
    print("\n== %s: n=%d q=%d sigma_s=%s sigma_e=%s" % (name, n, q, sigma_s, sigma_e))
    try:
        r = LWE.estimate(p)
        costs = {}
        for tag, c in r.items():
            costs[tag] = float(np.log2(float(c["rob"]))) if False else float(np.log2(float(c["rop"])))
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
    # ML-KEM: q=3329, CBD(2) -> sigma = 1.0
    estimate_exact("ML-KEM-512 (anchor)", 512, 3329, 2, 2)
    estimate_exact("ML-KEM-768 (anchor)", 768, 3329, 2, 2)
    estimate_exact("ML-KEM-1024 (anchor)", 1024, 3329, 2, 2)
    # FrodoKEM: q=2^15, noise sigma 2.65 / 2.30 (approximated as discrete Gaussian)
    estimate("FrodoKEM-640 (anchor, approx)", 640, 2**15, 2.65, 2.65)
    estimate("FrodoKEM-976 (anchor, approx)", 976, 2**15, 2.30, 2.30)

    print()
    print("#" * 76)
    print("# DUOLITH-KEM walls (current parameters)")
    print("#" * 76)
    estimate_exact("Wall-S / DUOLITH-128", 512, 7681, 3, 3)
    estimate_exact("Wall-S / DUOLITH-192", 768, 7681, 3, 3)
    estimate_exact("Wall-U / DUOLITH-128", 640, 2**16, 8, 8)
    estimate_exact("Wall-U / DUOLITH-192", 768, 2**16, 8, 8)
