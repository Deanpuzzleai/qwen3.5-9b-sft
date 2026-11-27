#!/usr/bin/env bash
# Download the locked base-model revision to shared storage and verify every file against the hashes recorded at lock time.
#
# Run from the repository root:
#   bash scripts/fetch_base_model.sh [DEST_DIR]
# Inputs : configs/eval/model.yaml (repo + revision), configs/eval/model_snapshot/ (FILES.sha256.txt with size + sha256 per
#          file, and byte-exact copies of the small config files); network access to huggingface.co; HF_TOKEN from .env if
#          present (the model is public; the token only raises rate limits). Needs one project venv (.venv/train or .venv/hle).
# Outputs: DEST_DIR (default /mnt/apex-shared/models/<repo name>@<revision>) holding all files of the revision,
#          DEST_DIR/fetch.log (this script's output), and DEST_DIR/.VERIFIED once every file matched. GPU VMs copy from
#          DEST_DIR instead of re-downloading.
set -euo pipefail
case "${1:-}" in
  -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  -*) echo "unknown option: $1 (the only argument is DEST_DIR; see --help)" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "too many arguments (the only argument is DEST_DIR)" >&2; exit 2; }
[[ -f configs/eval/model.yaml ]] || { echo "run from the repository root" >&2; exit 1; }
for PY in .venv/train/bin/python .venv/hle/bin/python; do [[ -x "$PY" ]] && break; done
[[ -x "$PY" ]] || { echo "no project venv found (run scripts/setup_eval_env.sh or scripts/gpu_setup_train_env.sh first)" >&2; exit 1; }
REPO=$("$PY" -c "import yaml; print(yaml.safe_load(open('configs/eval/model.yaml'))['model']['repo'])")
REV=$("$PY" -c "import yaml; print(yaml.safe_load(open('configs/eval/model.yaml'))['model']['revision'])")
DEST=${1:-/mnt/apex-shared/models/$(basename "$REPO")@${REV}}
HF_TOKEN=$( [[ -f .env ]] && { . ./.env >/dev/null 2>&1; printf '%s' "${HF_TOKEN:-}"; } || true ); export HF_TOKEN   # the only value read from .env
mkdir -p "$DEST"
rm -f "$DEST/.VERIFIED"   # a stale marker must not survive a failed re-verification
exec > >(tee -a "$DEST/fetch.log") 2>&1
echo "[fetch] $REPO @ $REV -> $DEST ($(date -u +%FT%TZ))"
"$PY" - "$REPO" "$REV" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:4]
print("[fetch] downloaded to", snapshot_download(repo_id=repo, revision=rev, local_dir=dest, max_workers=8))
PY
"$PY" - "$DEST" configs/eval/model_snapshot <<'PY'
import filecmp, hashlib, os, sys
dest, snapshot = sys.argv[1], sys.argv[2]
ok = bad = 0
for line in open(os.path.join(snapshot, "FILES.sha256.txt")):
    if line.startswith("#"):
        continue
    digest, size, name = line.rstrip("\n").split("  ", 2)
    path = os.path.join(dest, name)
    if not os.path.exists(path):
        print("[verify] MISSING", name); bad += 1; continue
    if str(os.path.getsize(path)) != size:
        print("[verify] SIZE MISMATCH", name, os.path.getsize(path), size); bad += 1; continue
    if len(digest) == 64:   # LFS files carry a sha256
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            print("[verify] SHA256 MISMATCH", name); bad += 1; continue
    else:   # small text files carry only a git blob id; compare byte-for-byte with the snapshot copy when we have one
        snap = os.path.join(snapshot, os.path.basename(name))
        if os.path.exists(snap) and not filecmp.cmp(path, snap, shallow=False):
            print("[verify] CONTENT MISMATCH vs configs/eval/model_snapshot:", name); bad += 1; continue
    ok += 1
print(f"[verify] {ok} files verified, {bad} problems")
sys.exit(1 if bad else 0)
PY
touch "$DEST/.VERIFIED"
echo "[fetch] VERIFIED $(du -sh "$DEST" | cut -f1) at $DEST ($(date -u +%FT%TZ))"
