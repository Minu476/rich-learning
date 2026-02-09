#!/usr/bin/env python3
"""Extract mean MFCC features from FSD50K NPZ files into a CSV for C# consumption."""

import numpy as np
import csv
import os
from pathlib import Path
from collections import Counter

# Paths — update these to point to your local FSD50K dataset
index_path = "path/to/fsd50k_full/index.csv"
gt_path = "path/to/FSD50K/FSD50K.ground_truth/dev.csv"
vocab_path = "path/to/FSD50K/FSD50K.ground_truth/vocabulary.csv"
out_path = "../data/fsd50k_features.csv"

# Load vocabulary
vocab = {}
with open(vocab_path) as f:
    for line in f:
        parts = line.strip().split(',')
        vocab[int(parts[0])] = parts[1]

# Task groups
instrument_labels = {
    'Acoustic_guitar', 'Bass_drum', 'Bass_guitar', 'Crash_cymbal', 'Cymbal',
    'Drum', 'Drum_kit', 'Electric_guitar', 'Guitar', 'Harmonica', 'Harp',
    'Organ', 'Piano', 'Snare_drum', 'Trumpet'
}
environment_labels = {
    'Car', 'Car_passing_by', 'Door', 'Engine', 'Engine_starting', 'Fire',
    'Knock', 'Rain', 'Raindrop', 'Siren', 'Thunder', 'Thunderstorm',
    'Traffic_noise_and_roadway_noise', 'Truck', 'Walk_and_footsteps', 'Water',
    'Wind'
}

# Load ground truth
gt = {}
with open(gt_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['fname']
        labels = [l.strip() for l in row['labels'].split(',')]
        gt[fname] = labels

# Load index
index_rows = []
with open(index_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row['source_path']
        wav_fname = Path(src).stem
        npz_path = row['saved_npz']
        index_rows.append((wav_fname, npz_path, row['file_id']))

print(f"Vocab: {len(vocab)} classes")
print(f"Ground truth: {len(gt)} entries")
print(f"Index: {len(index_rows)} files")

# Extract
os.makedirs(os.path.dirname(out_path), exist_ok=True)
instrument_count = 0
environment_count = 0
skipped = 0
rows_out = []

for wav_fname, npz_path, file_id in index_rows:
    if wav_fname not in gt:
        skipped += 1
        continue

    labels = gt[wav_fname]
    task = None
    primary_label = None
    for lbl in labels:
        if lbl in instrument_labels:
            task = 'A'
            primary_label = lbl
            break
        elif lbl in environment_labels:
            task = 'B'
            primary_label = lbl
            break

    if task is None:
        skipped += 1
        continue

    try:
        data = np.load(npz_path)
        feats = data[data.files[0]]
        mean_feats = feats.mean(axis=0)
        if task == 'A':
            instrument_count += 1
        else:
            environment_count += 1
        row = [file_id, wav_fname, task, primary_label] + [f"{x:.6f}" for x in mean_feats]
        rows_out.append(row)
    except Exception as e:
        skipped += 1

print(f"\nExtracted: {len(rows_out)} samples")
print(f"  Task A (instruments): {instrument_count}")
print(f"  Task B (environment): {environment_count}")
print(f"  Skipped: {skipped}")

# Write CSV
header = ['file_id', 'fname', 'task', 'label'] + [f'mfcc_{i}' for i in range(18)]
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows_out)

print(f"\nWrote {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")

label_counts = Counter(r[3] for r in rows_out)
print("\nLabel distribution:")
for lbl, cnt in label_counts.most_common():
    task = 'A' if lbl in instrument_labels else 'B'
    print(f"  [{task}] {lbl}: {cnt}")
