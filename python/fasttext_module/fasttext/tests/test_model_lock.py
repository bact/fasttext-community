"""Deadlock regression test for the per-model lock. Runs on every build.

Scenario:
- Readers call test_label(), which builds many Python objects.
- A writer calls set_matrices(), waiting on the model lock.
- A GC callback runs Python code, re-entering the same model and
  releasing the GIL.

Python <= 3.11 runs GC synchronously on allocation, so if Python objects
were built while holding the model lock, this deadlocks.
Runs in a subprocess so a deadlock fails the test instead of hanging it.
"""

import subprocess
import sys

import pytest

_SCRIPT = r"""
import gc, os, tempfile, threading, time
from fasttext.tests.helpers import build_supervised_model, get_random_data

# ~2000 labels (first word of each line), so test_label builds many objects.
# Pin thread=12: DenseMatrix::uniform() leaves part of the input matrix
# uninitialized when thread <= 10, so training can NaN.
data = get_random_data(20000, max_vocab_size=2000)
model = build_supervised_model(data, {"thread": 12, "dim": 20, "verbose": 0})

test = os.path.join(tempfile.mkdtemp(), "test.txt")
with open(test, "w") as f:
    f.write("__label__%s\n" % data[0])

inp, out = model.get_input_matrix(), model.get_output_matrix()
in_reader = threading.local()


def on_gc(phase, info):
    if phase == "start" and getattr(in_reader, "v", False):
        model.get_word_id("w1")  # re-enter the same model
        time.sleep(0.0005)  # release the GIL


gc.callbacks.append(on_gc)
gc.set_threshold(10)
stop = time.monotonic() + 3


def reader():
    in_reader.v = True
    while time.monotonic() < stop:
        model.test_label(test)


def writer():
    while time.monotonic() < stop:
        model.set_matrices(inp, out)


threads = [threading.Thread(target=reader) for _ in range(2)]
threads.append(threading.Thread(target=writer))
for t in threads:
    t.start()
for t in threads:
    t.join()
"""


def test_gc_callback_during_locked_call_does_not_deadlock():
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            timeout=120,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("deadlock: GC callback vs writer on the model lock")
    assert proc.returncode == 0, proc.stderr
