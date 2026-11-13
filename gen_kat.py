"""Generate the DUOLITH-KEM known-answer test vector file (deterministic)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duolith_kem as dk


def main(out_path):
    lines = []
    lines.append("DUOLITH-KEM v1.0 known-answer test vectors")
    lines.append("Deterministic: keygen seed material and encapsulation coins are")
    lines.append("derived with SHAKE-256 from the set name (see kat_material()).")
    lines.append("")
    for name in ("DUOLITH-128", "DUOLITH-192"):
        v = dk.kat_vectors(name)
        L = dk.lengths(name)
        lines.append("#" * 72)
        lines.append("set            = %s" % name)
        lines.append("keygen_seed    = %s   (%d bytes, hex)" % (v["seed"], 160))
        lines.append("encap_coins    = %s   (%d bytes, hex)" % (v["coins"], 64))
        lines.append("pk  (%5d B) = %s" % (L["pk"], v["pk"]))
        lines.append("sk  (%5d B) = %s" % (L["sk"], v["sk"]))
        lines.append("ct  (%5d B) = %s" % (L["ct"], v["ct"]))
        lines.append("ss  (%5d B) = %s" % (32, v["ss"]))
        # tamper vector: flip one bit in the Wall-U matrix block
        ct = bytearray(bytes.fromhex(v["ct"]))
        pos = L["ctS"] + 100
        ct[pos] ^= 0x40
        bad = bytes(ct)
        sk = bytes.fromhex(v["sk"])
        k_reject = dk.decapsulate(name, sk, bad)
        lines.append("ct_tampered    = flip bit 6 of byte %d of ct" % pos)
        lines.append("ss_reject      = %s   (decapsulate(sk, ct_tampered))" %
                      k_reject.hex())
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "KAT.txt")
