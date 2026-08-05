# Earlier attempts

Kept for reference, not maintained. These are standalone scripts, not
imports from `src/`.

- **v1_keras_baseline.py** — first pass, TensorFlow/Keras with
  `image_dataset_from_directory`. Same 4-block CNN idea, but a flat 0.25/0.5
  dropout schedule and a lower LR that made training slow to converge.
- **v2_pytorch_first_pass.py** — ported to PyTorch for more control over the
  training loop. Augmentation was too aggressive here (30° rotation +
  affine + random resized crop all stacked), which hurt more than it
  helped — validation accuracy plateaued lower than v1 despite a fancier
  pipeline.

`src/` is the version that came out of fixing both of those problems:
lighter, depth-scaled dropout, milder augmentation, gradient clipping, and
a cosine-annealing-with-warm-restarts schedule instead of a flat LR. That's
the one that actually produced the 67.6% test accuracy in the main README.
