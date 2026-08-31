#!/usr/bin/env bash
set -euo pipefail

##########################################
# Paths and basic config
##########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"

SPLITS_DIR="$BASE_DIR/splits"
SEQS_DIR="$BASE_DIR/sequences"
RAW_OUT_DIR="$BASE_DIR/pbsim_raw"
ERROR_PROFILE_DIR="$BASE_DIR/error_profiles"

SPLITS=("train" "id_novel_genus" "ood_novel_family" "ood_nonbacterial")

PBSIM_BIN="${PBSIM_BIN:-pbsim}"  # or pbsim3 if that's your binary name
QSHMM_MODEL="${QSHMM_MODEL:-$ERROR_PROFILE_DIR/QSHMM-RSII.model}"

DEPTH="${DEPTH:-1}"
LENGTH_MIN="${LENGTH_MIN:-10000}"
LENGTH_MAX="${LENGTH_MAX:-11000}"

MAX_SPECIES_PER_SPLIT="${MAX_SPECIES_PER_SPLIT:-}"

##########################################
# Helper functions
##########################################

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

run_pbsim_for_species() {
  local split="$1"
  local species="$2"

  local species_dir="$SEQS_DIR/$species"
  if [[ ! -d "$species_dir" ]]; then
    log "WARNING: species dir not found: $species_dir (skipping)"
    return
  fi

  # Collect fasta-like files
  shopt -s nullglob
  local fasta_files=("$species_dir"/*.fa "$species_dir"/*.fna "$species_dir"/*.fasta)
  shopt -u nullglob

  if [[ "${#fasta_files[@]}" -eq 0 ]]; then
    log "WARNING: no FASTA files in $species_dir (skipping)"
    return
  fi

  local split_out_dir="$RAW_OUT_DIR/$split/$species"
  mkdir -p "$split_out_dir"

  # Concatenate all FASTAs into a single temp file
  local tmp_fasta="$split_out_dir/${species}.combined.fasta"
  cat "${fasta_files[@]}" > "$tmp_fasta"

  log "Running PBSIM for split=$split species=$species"
  log "  FASTA : $tmp_fasta"
  log "  Outdir: $split_out_dir"

  (
    cd "$split_out_dir"

    "$PBSIM_BIN" --strategy wgs \
                 --method qshmm \
                 --qshmm "$QSHMM_MODEL" \
                 --genome "$tmp_fasta" \
                 --length-min "$LENGTH_MIN" \
                 --length-max "$LENGTH_MAX" \
                 --depth "$DEPTH" \
                 --prefix "$species"
  )

  # Clean up intermediates: keep only FASTQ reads (and dir structure)
  log "Cleaning intermediates for split=$split species=$species"
  rm -f "$tmp_fasta"               # combined fasta we created
  rm -f "$split_out_dir"/*.ref     # PBSIM reference copy
  rm -f "$split_out_dir"/*.maf.gz  # alignment files (if you don't need them)

  log "Finished PBSIM for split=$split species=$species"
}

##########################################
# Main
##########################################

log "Base dir           : $BASE_DIR"
log "Splits dir         : $SPLITS_DIR"
log "Sequences dir      : $SEQS_DIR"
log "Raw PBSIM out dir  : $RAW_OUT_DIR"
log "Error profile dir  : $ERROR_PROFILE_DIR"
log "PBSIM binary       : $PBSIM_BIN"
log "QSHMM model        : $QSHMM_MODEL"
log "Depth              : $DEPTH"
log "Length range (bp)  : $LENGTH_MIN - $LENGTH_MAX"

mkdir -p "$RAW_OUT_DIR"

for split in "${SPLITS[@]}"; do
  local_list="$SPLITS_DIR/${split}_species.txt"
  if [[ ! -f "$local_list" ]]; then
    log "WARNING: species list not found for split=$split: $local_list (skipping)"
    continue
  fi

  log "=== Split: $split ==="
  count=0

  while read -r species; do
    [[ -z "$species" ]] && continue

    run_pbsim_for_species "$split" "$species"
    count=$((count + 1))

    if [[ -n "$MAX_SPECIES_PER_SPLIT" && "$count" -ge "$MAX_SPECIES_PER_SPLIT" ]]; then
      log "Reached MAX_SPECIES_PER_SPLIT=$MAX_SPECIES_PER_SPLIT for split=$split, stopping early."
      break
    fi
  done < "$local_list"

  log "Completed split=$split (#species simulated: $count)"
done

log "All splits done. Raw PBSIM outputs are under: $RAW_OUT_DIR"
