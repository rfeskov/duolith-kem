# Errata for DUOLITH-KEM Specification v1.0 (PDF)

The PDF is not regenerated in this change. The items below record where the
v1.0 text disagrees with the reference implementation, with its own tables, or
with published anchor parameters.

## Implementation vs. specification

1. **Wall-U E'' noise (Section 6, Encrypt).** The spec samples
   `E'' <- CBD(8)^64`. The v1.0 code requested only `2*t*t/8 = 16` bytes from
   the PRF where CBD(8) over 64 entries needs `2*eta*t*t/8 = 128` bytes.
   `np.unpackbits` zero-pads silently, so only 8 of the 64 entries received
   noise and the other 56 were exactly zero. Fixed in `duolith_kem.py`;
   `cbd_from_stream` now raises on an under-sized stream. **The KAT vectors
   change as a result** (`KAT.txt` regenerated). The failure analysis already
   modelled the full CBD(8) E'', so its bounds are unaffected.

2. **Module docstring.** Ciphertext widths were listed as u 11 / v 5 / U 13 /
   C 12 bits; the parameter sets (and the spec's Table 5) use 12 / 6 / 14 / 13.
   The docstring also said FO re-encryption randomness is derived from H(ct);
   both the code and Section 10.3 use H(pk). Fixed.

## Security-estimate harness (Section 10.1, Table 8)

3. `security_estimates.py` used `np` in `estimate()` without importing numpy.
   The resulting `NameError` was caught and printed as "estimate failed", so
   the FrodoKEM rows of Table 8 cannot have been produced by the committed
   script. It also hard-coded `/tmp/le` and `/home/z/my-project/...` paths.
   Fixed: numpy is imported, the estimator path comes from
   `LATTICE_ESTIMATOR`, and the bundled `all.py` shim is registered as
   `sage.all` automatically.

4. **Anchor parameters.** Table 8 should be re-run with corrected anchors:
   - ML-KEM-512 key-generation noise is CBD(eta1) with eta1 = 3 (FIPS 203),
     not CBD(2). ML-KEM-768/1024 use eta1 = 2, as before.
   - FrodoKEM-640 uses sigma = 2.8 (FrodoKEM specification), not 2.65.
   - FrodoKEM-976 uses q = 2^16, not 2^15.

## Statements that conflict with the spec's own numbers

5. **Section 10.3:** "both DUOLITH-128 walls individually meet or exceed the
   ML-KEM-512 anchor". Table 8 lists Wall-S / DUOLITH-128 at 2^127.5, below
   the ML-KEM-512 anchor at 2^132.8 (and that anchor was itself computed with
   too-small noise, see item 4).

6. **Section 10.2:** "if one wall were annihilated ... security would degrade
   to the other wall (144.6 or 173.7 bits)". For DUOLITH-128 the
   single-wall floor is the weaker wall, Wall-S at 127.5 bits, not 144.6.

7. **Section 3:** "a breakthrough on plain LWE would leave Wall-S intact".
   Module-LWE is a special case of LWE, so an attack on general LWE (or a
   generic improvement in lattice reduction) applies to both walls. The
   combiner only hedges against attacks that exploit ring/module structure.

## Design concerns (not fixed here)

8. **Decapsulation failure rate.** Table 5 gives <= 2^-107.1 (DUOLITH-128)
   and <= 2^-89.1 (DUOLITH-192), against a stated design target of 2^-64.
   For an IND-CCA KEM, failure rates this far above the claimed security
   level leave room for failure-boosting attacks; ML-KEM and FrodoKEM target
   far lower rates. Both bounds come from Wall-U (Wall-S is below 2^-500),
   and DUOLITH-192 fails more often than DUOLITH-128.

9. `SCALED-S` and `SCALED-U` in `duolith_kem.py` are identical parameter sets.

10. The cover page states a 28-test suite; the suite now has 29 checks.
