"""Concurrency tests for the free-threaded (cp314t, no-GIL) build.

Best-effort, probabilistic stress tests:
- Passing does not prove absence of data races -- only ThreadSanitizer does.
- Skipped unless the GIL is actually off (`PYTHON_GIL=0` / `-X gil=0`) on a
  `cp314t`-or-later interpreter.
"""

import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from .helpers import build_supervised_model, get_random_data

_GIL_DISABLED = bool(sysconfig.get_config_var("Py_GIL_DISABLED")) and not getattr(
    sys, "_is_gil_enabled", lambda: True
)()

pytestmark = pytest.mark.skipif(
    not _GIL_DISABLED,
    reason="requires a free-threaded (cp314t+) interpreter with the GIL actually off",
)

N_THREADS = 16
N_ITERATIONS = 20


def _build_model():
    # Pin thread=12: DenseMatrix::uniform() leaves part of the input matrix
    # uninitialized when thread <= 10, so training can NaN and the test
    # skips. The helpers default is thread=1.
    return build_supervised_model(
        get_random_data(200, max_vocab_size=200), {"thread": 12}
    )


def test_gil_stays_disabled_after_import():
    """Confirms py::mod_gil_not_used() worked: GIL stays off after import."""
    assert sysconfig.get_config_var("Py_GIL_DISABLED")
    assert sys._is_gil_enabled() is False


def test_concurrent_reads_are_safe_and_consistent():
    model = _build_model()
    words = model.get_words()
    assert words, "expected a non-empty vocabulary"

    reference_vectors = {w: model.get_word_vector(w) for w in words[:10]}

    def worker():
        for _ in range(N_ITERATIONS):
            for word in words[:10]:
                vec = model.get_word_vector(word)
                assert (vec == reference_vectors[word]).all()
            model.predict(words[0])
            model.get_nearest_neighbors(words[0], k=3)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [pool.submit(worker) for _ in range(N_THREADS)]
        for future in as_completed(futures):
            future.result()


def test_concurrent_train_vs_read_is_safe():
    """Writer vs reader lock split (exclusive vs shared).

    - Writer: loops set_matrices() on a shared model.
    - Readers: concurrently call predict / get_word_vector / get_nearest_neighbors.
    - set_matrices also resets the getNN/getAnalogies word-vectors cache
      (see test_concurrent_first_use_of_nearest_neighbors_is_safe).
    - Uses set_matrices, not train()/quantize(): quantize() is one-shot per
      model, so it can't drive a repeatable writer loop.
    """
    model = _build_model()
    words = model.get_words()
    assert words

    input_matrix = model.get_input_matrix()
    output_matrix = model.get_output_matrix()

    stop = threading.Event()
    errors = []

    def writer():
        try:
            for _ in range(N_ITERATIONS):
                if stop.is_set():
                    return
                model.set_matrices(input_matrix, output_matrix)
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                model.predict(words[0])
                model.get_word_vector(words[0])
                model.get_nearest_neighbors(words[0], k=3)
        except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [pool.submit(writer)]
        futures += [pool.submit(reader) for _ in range(N_THREADS - 1)]
        for future in as_completed(futures):
            future.result()

    assert not errors, errors


def test_concurrent_first_use_of_nearest_neighbors_is_safe():
    """Targets the lazyComputeWordVectors() check-then-act race (src/fasttext.cc).

    Hammers getNN/getAnalogies concurrently, first call ever, on fresh models.
    """
    for _ in range(5):
        model = _build_model()
        words = model.get_words()
        if len(words) < 3:
            continue

        barrier = threading.Barrier(N_THREADS)
        results = []
        errors = []

        def worker():
            try:
                barrier.wait()
                if threading.current_thread().name.endswith("0"):
                    results.append(model.get_analogies(words[0], words[1], words[2]))
                else:
                    results.append(model.get_nearest_neighbors(words[0], k=3))
            except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, name=f"nn-worker-{i}")
            for i in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(results) == N_THREADS
