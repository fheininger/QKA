"""Mirror ROC comparison plots where the quantum kernel beat the classical RBF baseline.

The AUCs of a comparison plot exist only as text inside the PNG legend, so a plot that
lands in results/roc_curves/ carries no machine-readable record of whether the quantum
kernel actually won. record_roc_comparison() is called right after plt.savefig(), while
both AUCs are still floats, and mirrors the winners into a subfolder with a manifest.
"""
import csv
import os
import shutil
from datetime import datetime

WIN_SUBDIR = "quantum_beats_rbf"
MANIFEST = "manifest.csv"
FIELDS = ["file", "quantum_auc", "classical_rbf_auc", "delta",
          "ci_lo", "ci_hi", "beats_ci", "dataset", "model", "qubits", "recorded"]


def _upsert(manifest_path, row):
    """Rewrite the manifest with `row` replacing any existing row for the same file.

    Older manifests carry only the first four FIELDS; missing keys are backfilled with
    "" so a legacy file is migrated rather than rejected.
    """
    rows = []
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("file") != row["file"]]
    rows.append(row)
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        # not `or ""`: that would blank a legitimate False in beats_ci
        w.writerows({f: "" if r.get(f) is None else r.get(f, "") for f in FIELDS}
                    for r in rows)


def record_roc_comparison(path, auc_q, auc_c, ci=None, dataset="", model="", qubits=""):
    """Copy `path` into <its dir>/quantum_beats_rbf/ when the quantum AUC beat the RBF one.

    path    ROC comparison png just written by plt.savefig
    auc_q   quantum AUC, auc_c classical RBF AUC
    ci      optional (lo, hi) bootstrap 95% CI of the quantum AUC; records whether the
            RBF AUC falls below the lower bound, which the raw win alone does not tell you

    Nothing is ever deleted: a re-run that no longer wins leaves the existing copy and its
    manifest row in place and only warns, so the contradiction shows up in the run log.
    """
    win_dir = os.path.join(os.path.dirname(os.path.abspath(path)), WIN_SUBDIR)
    base = os.path.basename(path)

    if auc_q <= auc_c:
        if os.path.exists(os.path.join(win_dir, base)):
            print(f"⚠️ {base}: quantum {auc_q:.3f} no longer beats RBF {auc_c:.3f}, "
                  f"but an earlier winning copy is kept in {WIN_SUBDIR}/ (superseded)")
        return False

    os.makedirs(win_dir, exist_ok=True)
    shutil.copy2(path, os.path.join(win_dir, base))
    lo, hi = (ci if ci is not None else ("", ""))
    _upsert(os.path.join(win_dir, MANIFEST), {
        "file": base,
        "quantum_auc": round(auc_q, 4),
        "classical_rbf_auc": round(auc_c, 4),
        "delta": round(auc_q - auc_c, 4),
        "ci_lo": round(lo, 4) if ci is not None else "",
        "ci_hi": round(hi, 4) if ci is not None else "",
        "beats_ci": (auc_c < lo) if ci is not None else "",
        "dataset": dataset,
        "model": model,
        "qubits": qubits,
        "recorded": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    print(f"✅ {base}: quantum {auc_q:.3f} > RBF {auc_c:.3f} → copied to {WIN_SUBDIR}/")
    return True
