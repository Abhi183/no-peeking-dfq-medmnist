# Training provenance for the 12 checkpoints

The training script prints progress to the terminal and never wrote a log
file, and the checkpoint dictionaries do not record the epoch budget or
learning rate. This directory collects the documentary evidence that does
exist, recovered on 2026-10-07 from the session database of the coding
assistant that launched the runs.

## What the records say

Both training runs were launched as background commands on 2026-06-13 from
`~/medcompress`, using `dfq/train.py` at commit `c0a806b` (committed
04:31:24 UTC that day; the file has not changed since).

| UTC time | Record | Content |
|---|---|---|
| 04:31:49 | launch observation #2431 | "12-epoch training on dermamnist, pneumoniamnist, bloodmnist, and pathmnist with seed 42", "64x64 resolution" |
| 04:32:31 to 04:47:20 | checkpoint mtimes | four seed-42 checkpoints written in that order |
| 04:47:22 | completion notice #1848 | `Background command "Train all 4 MedMNIST datasets seed 42, 12 epochs" completed (exit code 0)` |
| 22:11:45 | launch observation #2446 | "2 random seeds (123, 456)", "12 epochs with 64x64 image size" |
| 22:12:26 to 22:39:59 | checkpoint mtimes | eight seed-123 and seed-456 checkpoints |
| 22:40:02 | completion notice #1853 | `Background command "Train all 4 datasets for seeds 123 and 456" completed (exit code 0)` |

The per-dataset test metrics recorded in observations #2432, #2435, and #2449
match the `test` dictionaries stored inside the checkpoints to the printed
precision (for example DermaMNIST seed 42: accuracy 0.7451, AUC 0.9117 in
both). The recorded training times (56 s, 38 s, 92 s, 675 s for the seed-42
run) match the gaps between consecutive checkpoint mtimes.

## Reconstructed commands

```bash
python dfq/train.py --datasets dermamnist pneumoniamnist bloodmnist pathmnist --seeds 42 --epochs 12 --size 64
python dfq/train.py --datasets dermamnist pneumoniamnist bloodmnist pathmnist --seeds 123 456 --epochs 12 --size 64
```

The exact command strings were not saved. The flags above are the ones the
launch records name. Neither record mentions `--lr`, and the script default at
commit `c0a806b` was `1e-3`. We therefore treat the learning rate as the
default, with the caveat that an explicit override would have left no trace.

## What this settles and what it does not

Settled: the epoch budget was 12, image size 64, seeds 42, 123, 456, and the
script was the committed version. This is consistent with the batch-norm
counter analysis in `../storage_study/training_provenance.json`, which places
every selected checkpoint at epoch 6 to 12.

Not settled by a primary log: the learning rate. It is inferred from the
script default and the absence of an override in the launch records.

## Files

- `checkpoint_mtimes.csv`: modification time (UTC) and SHA-256 of each checkpoint.
- `train_py_at_commit_c0a806b.py`: the training script as committed, for
  comparison with `code/train.py`.

The session records themselves (launch observations and completion notices)
live in the first author's local assistant database and are not redistributed
here. They can be provided to reviewers on request. The table above quotes the
relevant lines.

## Going forward

`code/train.py` now writes `checkpoints/<dataset>_seed<seed>.train.json` with
the parsed arguments, per-epoch validation metrics, package versions, device,
and wall-clock time, and stores the arguments inside the checkpoint under
`train_args`. Future runs will not depend on recovered session records.
