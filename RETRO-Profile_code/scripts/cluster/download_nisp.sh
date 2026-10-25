#!/bin/bash
# Download the NISP dataset onto the GPU cluster.  🔵 RUN ON THE the GPU cluster LOGIN NODE.
#
# Login nodes have internet; compute nodes do not (which is why HF_HUB_OFFLINE=1
# is set in the training jobs). This also sidesteps the data-transfer node, which
# refuses connections from the user's Mac.
#
# WHERE THE DATA LIVES. NISP is committed straight into its GitHub repo as split
# archives, not as release assets:
#
#     <Lang>_master/<session>/RECS.tar.gz.aa ... .au      (~95 MB per part)
#
# with sessions <Lang> (native speech) and English_<Lang>. Hindi's native session
# alone is 21 parts, about 2.1 GB. Five languages x two sessions is roughly 20 GB
# compressed.
#
# DO NOT `git clone` THIS REPO. Git keeps every part twice -- once in .git, once
# in the working tree -- so a clone costs double and pulls all ten archives
# whether you want them or not. This script fetches raw files instead.
#
# RULES. Everything lands under ~/<project>/<user>/ (R1). Never <scratch>, which
# was at 109% of quota (R2). Check `user_info` before starting: you need ~20 GB
# for the archives and roughly the same again once extracted.
#
# EXTRACTION IS NOT DONE HERE. Unpacking ~20 GB is heavy I/O and the GPU cluster discourages
# that on a login node. This script only downloads and reassembles; it prints the
# extraction command to run as a batch job on the `comp` partition.
#
# USAGE
#   bash scripts/cluster/download_nisp.sh --list                 # show sizes, download nothing
#   bash scripts/cluster/download_nisp.sh --english              # English sessions only (~10 GB)
#   bash scripts/cluster/download_nisp.sh                        # everything (~20 GB)
#   LANGS="Hindi Tamil" bash scripts/cluster/download_nisp.sh    # a subset
#
# Re-running is safe: completed parts are skipped, partial ones resume (wget -c).

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
DEST="${DEST:-$HOME/<project>/${USER_NS}/nisp}"
REPO="iiscleap/NISP-Dataset"
BRANCH="${BRANCH:-master}"
LANGS="${LANGS:-Hindi Kannada Malayalam Tamil Telugu}"

MODE="all"
case "${1:-}" in
  --list)    MODE="list" ;;
  --english) MODE="english" ;;
  "")        ;;
  *) echo "unknown option: $1"; exit 1 ;;
esac

# Refuse to write outside our own namespace (R1); setup_env.sh has the same guard.
case "$(readlink -f "$(dirname "${DEST}")")" in
  *"/${USER_NS}"*) ;;
  *) echo "REFUSING: ${DEST} resolves outside ${USER_NS}/. Set DEST explicitly."; exit 1 ;;
esac
case "${DEST}" in
  *scratch2*) echo "REFUSING: <scratch> was at 109% of quota (R2)."; exit 1 ;;
esac

mkdir -p "${DEST}"
echo "destination: ${DEST}"
echo

# Sessions per language: native speech, and the English recordings of the same
# speakers. English-only is the cheaper starting point and is more comparable to
# TIMIT, since every speaker is then in one language.
sessions_for() {
  local lang="$1"
  if [ "${MODE}" = "english" ]; then echo "English_${lang}"; else echo "${lang} English_${lang}"; fi
}

# List a directory's parts via the GitHub contents API. Parsed with python3
# (present on the login node) so this needs no jq.
list_parts() {
  local path="$1"
  python3 - "$path" <<'PY'
import json, sys, urllib.request
path = sys.argv[1]
url = f"https://api.github.com/repos/iiscleap/NISP-Dataset/contents/{path}"
try:
    with urllib.request.urlopen(url, timeout=60) as r:
        items = json.load(r)
except Exception as e:                       # noqa: BLE001 - reported, not raised
    print(f"__ERROR__ {e}", file=sys.stderr)
    sys.exit(1)
if not isinstance(items, list):
    sys.exit(1)
for it in items:
    if it.get("type") == "file" and it["name"].startswith("RECS.tar.gz."):
        print(f"{it['name']}\t{it['size']}\t{it.get('download_url') or ''}")
PY
}

total_bytes=0
for lang in ${LANGS}; do
  for sess in $(sessions_for "${lang}"); do
    path="${lang}_master/${sess}"
    parts="$(list_parts "${path}" || true)"
    if [ -z "${parts}" ]; then
      echo "  ${path}: no parts found (skipping)"
      continue
    fi
    n=$(echo "${parts}" | wc -l | tr -d ' ')
    bytes=$(echo "${parts}" | awk -F'\t' '{s+=$2} END {print s+0}')
    total_bytes=$(( total_bytes + bytes ))
    printf '  %-34s %2s parts  %6.2f GB\n' "${path}" "${n}" \
      "$(echo "${bytes}" | awk '{print $1/1073741824}')"

    [ "${MODE}" = "list" ] && continue

    out="${DEST}/${path}"
    mkdir -p "${out}"
    echo "${parts}" | while IFS=$'\t' read -r name size url; do
      target="${out}/${name}"
      # Skip parts already at their full size; resume anything short.
      if [ -f "${target}" ] && [ "$(stat -c%s "${target}" 2>/dev/null || echo 0)" = "${size}" ]; then
        continue
      fi
      # NOT `wget -c -O`. With -O, wget cannot tell how much of the named file
      # belongs to this URL, so -c APPENDS instead of resuming -- any retry or
      # re-run silently doubles the file and the reassembled archive is corrupt
      # ("tar: This does not look like a tar archive"). Download to a .part file
      # and rename only on success, so an interrupted transfer is simply redone.
      rm -f "${target}.part"
      if wget -q --show-progress -O "${target}.part" "${url}"; then
        got=$(stat -c%s "${target}.part" 2>/dev/null || echo 0)
        if [ "${got}" = "${size}" ]; then
          mv -f "${target}.part" "${target}"
        else
          echo "    SIZE MISMATCH ${name}: got ${got}, expected ${size} -- discarding"
          rm -f "${target}.part"
        fi
      else
        echo "    DOWNLOAD FAILED ${name}"
        rm -f "${target}.part"
      fi
    done

    # Reassemble. `cat` in shell glob order is the correct order here: the parts
    # are named .aa .ab .ac ... so lexical order IS split order.
    # Reassemble, then VERIFY. A silently corrupt archive is worse than a
    # missing one: it fails hours later inside a batch job, as it did on
    # 2026-08-31 when every part had been doubled by `wget -c -O`.
    archive="${out}/RECS.tar.gz"
    if [ ! -f "${archive}" ]; then
      have=$(ls "${out}"/RECS.tar.gz.a* 2>/dev/null | wc -l | tr -d ' ')
      if [ "${have}" != "${n}" ]; then
        echo "    INCOMPLETE: ${have} of ${n} parts present -- not reassembling"
        continue
      fi
      cat "${out}"/RECS.tar.gz.a* > "${archive}"
      got=$(stat -c%s "${archive}" 2>/dev/null || echo 0)
      if [ "${got}" != "${bytes}" ]; then
        echo "    SIZE MISMATCH after reassembly: ${got} vs expected ${bytes} -- removing"
        rm -f "${archive}"
        continue
      fi
      if ! gzip -t "${archive}" 2>/dev/null; then
        echo "    NOT A VALID GZIP: ${archive} -- removing"
        rm -f "${archive}"
        continue
      fi
      echo "    reassembled + verified $(du -h "${archive}" | cut -f1) -> ${archive}"
    fi
  done
done

# Report what is actually on disk, not the API's byte sum. The summed figure
# under-reported by 2x on the first real run (said 15.75 GB, the archives were
# 32 GB), and a wrong number here leads to a wrong quota decision -- which on a
# shared project (rule R1) is not a harmless mistake. du is the ground truth.
if [ "${MODE}" != "list" ]; then
  printf '\non disk: %s\n' "$(du -sh "${DEST}" 2>/dev/null | cut -f1)"
  echo "(API byte-sum estimate was $(echo "${total_bytes}" | awk '{printf "%.2f", $1/1073741824}') GB -- trust du, not this)"
else
  printf '\nestimated total: %.2f GB (API byte sum; verify with du after downloading)\n' \
    "$(echo "${total_bytes}" | awk '{print $1/1073741824}')"
fi

if [ "${MODE}" = "list" ]; then
  echo "(--list: nothing downloaded)"
  exit 0
fi

cat <<EOF

Downloaded and reassembled. Do NOT extract on the login node -- submit this
instead (CPU partition, no GPU needed):

  cd ${DEST}
  find . -name 'RECS.tar.gz' | while read -r a; do
    echo "tar -xzf \$a -C \$(dirname \$a)"
  done > extract_cmds.sh

  sbatch --account=<project> --partition=comp --time=02:00:00 --cpus-per-task=4 \\
         --mem=8G --wrap 'bash extract_cmds.sh'

Once extracted, the speaker labels (height, weight, age, gender, mother tongue)
are in the repo root file total_spkrinfo.list -- fetch it with:

  wget -q -O ${DEST}/total_spkrinfo.list \\
    https://raw.githubusercontent.com/${REPO}/${BRANCH}/total_spkrinfo.list
  wget -q -O ${DEST}/train_spkrID https://raw.githubusercontent.com/${REPO}/${BRANCH}/train_spkrID
  wget -q -O ${DEST}/test_spkrID  https://raw.githubusercontent.com/${REPO}/${BRANCH}/test_spkrID

NOTE: scripts/prepare_nisp.py does NOT match this layout. It expects
<raw>/<language>/audio/<speaker>/*.wav plus a per-language metadata.csv; the
real release is <Lang>_master/<session>/ with one total_spkrinfo.list at the
root and filenames like Hin_0001_Eng_f_0000.wav. It needs rewriting against the
actual tree before the manifests will build.
EOF
