"""The training run record must capture what the checkpoint itself does not."""
import json

from train import build_run_record


def _record(**overrides):
    epoch_log = [
        {"epoch": 1, "val_acc": 0.5, "val_auc": 0.70, "val_f1_macro": 0.4, "selected": True},
        {"epoch": 2, "val_acc": 0.6, "val_auc": 0.80, "val_f1_macro": 0.5, "selected": True},
        {"epoch": 3, "val_acc": 0.6, "val_auc": 0.79, "val_f1_macro": 0.5, "selected": False},
    ]
    kwargs = dict(
        dataset="dermamnist", seed=42, epochs=3, size=64, lr=1e-3,
        batch_size=128, weight_decay=1e-4, device="mps",
        epoch_log=epoch_log, test={"acc": 0.7, "auc": 0.9, "f1_macro": 0.5},
        train_secs=56.04, started_utc="2026-06-13T04:31:35+00:00",
    )
    kwargs.update(overrides)
    return build_run_record(**kwargs)


def test_record_stores_epoch_budget_and_learning_rate():
    rec = _record()
    assert rec["args"]["epochs"] == 3
    assert rec["args"]["lr"] == 1e-3
    assert rec["args"]["batch_size"] == 128


def test_selected_epoch_is_last_epoch_that_improved_val_auc():
    assert _record()["selected_epoch"] == 2
    assert _record(epoch_log=[])["selected_epoch"] is None


def test_record_is_json_serialisable_and_names_versions():
    rec = _record()
    text = json.dumps(rec)
    assert "torch" in rec["versions"] and "numpy" in rec["versions"]
    assert rec["epochs_run"] == 3
    assert rec["train_secs"] == 56.0
    assert json.loads(text)["dataset"] == "dermamnist"
