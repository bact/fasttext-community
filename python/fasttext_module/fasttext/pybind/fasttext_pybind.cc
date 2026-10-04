/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <args.h>
#include <autotune.h>
#include <densematrix.h>
#include <fasttext.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <real.h>
#include <vector.h>
#include <cmath>
#include <iterator>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

using namespace pybind11::literals;
namespace py = pybind11;

// FastText + reader/writer lock, for concurrent use from Python threads.
// - shared lock: read-only methods (predict, getNN, ...)
// - exclusive lock: methods that replace state (train, loadModel,
//   quantize, setMatrices)
// pybind layer only: the CLI and WebAssembly builds are single-threaded.
//
// Rules:
// 1. Lock via readLock()/writeLock(): they release the GIL while blocked.
// 2. Under the lock, do C++ work only. Build Python objects after.
// Why: on Python <= 3.11, allocating a Python object can run GC, i.e.
// arbitrary Python code. That code may re-enter this non-recursive lock,
// or hand the GIL to a thread that then blocks on the lock: deadlock.
struct FastTextHandle
{
  fasttext::FastText ft;
  std::shared_mutex mu;
};

// Waits on `mu` without the GIL. On free-threaded builds this also
// detaches the thread, so a waiter can't stall a stop-the-world pause.
// Caller must hold the GIL.
template <typename Lock>
Lock lockReleasingGil(std::shared_mutex &mu)
{
  Lock lock(mu, std::try_to_lock);
  if (!lock.owns_lock())
  {
    py::gil_scoped_release release;
    lock.lock();
  }
  return lock;
}

std::shared_lock<std::shared_mutex> readLock(FastTextHandle &w)
{
  return lockReleasingGil<std::shared_lock<std::shared_mutex>>(w.mu);
}

std::unique_lock<std::shared_mutex> writeLock(FastTextHandle &w)
{
  return lockReleasingGil<std::unique_lock<std::shared_mutex>>(w.mu);
}

py::str castToPythonString(const std::string &s, const char *onUnicodeError)
{
  PyObject *handle = PyUnicode_DecodeUTF8(s.data(), s.length(), onUnicodeError);
  if (!handle)
  {
    throw py::error_already_set();
  }

  // py::str's constructor from a PyObject assumes the string has been encoded
  // for python 2 and not encoded for python 3 :
  // https://github.com/pybind/pybind11/blob/ccbe68b084806dece5863437a7dc93de20bd9b15/include/pybind11/pytypes.h#L930
#if PY_MAJOR_VERSION < 3
  PyObject *handle_encoded =
      PyUnicode_AsEncodedString(handle, "utf-8", onUnicodeError);
  Py_DECREF(handle);
  handle = handle_encoded;
#endif

  py::str handle_str = py::str(handle);
  Py_DECREF(handle);
  return handle_str;
}

std::vector<std::pair<fasttext::real, py::str>> castToPythonString(
    const std::vector<std::pair<fasttext::real, std::string>> &predictions,
    const char *onUnicodeError)
{
  std::vector<std::pair<fasttext::real, py::str>> transformedPredictions;

  for (const auto &prediction : predictions)
  {
    transformedPredictions.emplace_back(
        prediction.first,
        castToPythonString(prediction.second, onUnicodeError));
  }

  return transformedPredictions;
}

std::vector<py::str> castToPythonString(
    const std::vector<std::string> &strings,
    const char *onUnicodeError)
{
  std::vector<py::str> transformed;
  for (const auto &s : strings)
  {
    transformed.push_back(castToPythonString(s, onUnicodeError));
  }
  return transformed;
}

// Split `text` into (words, labels). C++ only: safe under the model lock.
std::pair<std::vector<std::string>, std::vector<std::string>> getLineTokens(
    const fasttext::FastText &m,
    const std::string &text)
{
  std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
  std::stringstream ioss(text);
  std::string token;
  std::vector<std::string> words;
  std::vector<std::string> labels;
  while (d->readWord(ioss, token))
  {
    uint32_t h = d->hash(token);
    int32_t wid = d->getId(token, h);
    fasttext::entry_type type = wid < 0 ? d->getType(token) : d->getType(wid);

    if (type == fasttext::entry_type::word)
    {
      words.push_back(token);
      // Labels must not be OOV!
    }
    else if (type == fasttext::entry_type::label && wid >= 0)
    {
      labels.push_back(token);
    }
    if (token == fasttext::Dictionary::EOS)
    {
      break;
    }
  }
  return {words, labels};
}

PYBIND11_MODULE(fasttext_pybind, m, py::mod_gil_not_used())
{
  py::class_<fasttext::Args>(m, "args")
      .def(py::init<>())
      .def_readwrite("input", &fasttext::Args::input)
      .def_readwrite("output", &fasttext::Args::output)
      .def_readwrite("lr", &fasttext::Args::lr)
      .def_readwrite("lrUpdateRate", &fasttext::Args::lrUpdateRate)
      .def_readwrite("dim", &fasttext::Args::dim)
      .def_readwrite("ws", &fasttext::Args::ws)
      .def_readwrite("epoch", &fasttext::Args::epoch)
      .def_readwrite("minCount", &fasttext::Args::minCount)
      .def_readwrite("minCountLabel", &fasttext::Args::minCountLabel)
      .def_readwrite("neg", &fasttext::Args::neg)
      .def_readwrite("wordNgrams", &fasttext::Args::wordNgrams)
      .def_readwrite("loss", &fasttext::Args::loss)
      .def_readwrite("model", &fasttext::Args::model)
      .def_readwrite("bucket", &fasttext::Args::bucket)
      .def_readwrite("minn", &fasttext::Args::minn)
      .def_readwrite("maxn", &fasttext::Args::maxn)
      .def_readwrite("thread", &fasttext::Args::thread)
      .def_readwrite("t", &fasttext::Args::t)
      .def_readwrite("label", &fasttext::Args::label)
      .def_readwrite("verbose", &fasttext::Args::verbose)
      .def_readwrite("pretrainedVectors", &fasttext::Args::pretrainedVectors)
      .def_readwrite("saveOutput", &fasttext::Args::saveOutput)
      .def_readwrite("seed", &fasttext::Args::seed)

      .def_readwrite("qout", &fasttext::Args::qout)
      .def_readwrite("retrain", &fasttext::Args::retrain)
      .def_readwrite("qnorm", &fasttext::Args::qnorm)
      .def_readwrite("cutoff", &fasttext::Args::cutoff)
      .def_readwrite("dsub", &fasttext::Args::dsub)

      .def_readwrite(
          "autotuneValidationFile", &fasttext::Args::autotuneValidationFile)
      .def_readwrite("autotuneMetric", &fasttext::Args::autotuneMetric)
      .def_readwrite(
          "autotunePredictions", &fasttext::Args::autotunePredictions)
      .def_readwrite("autotuneDuration", &fasttext::Args::autotuneDuration)
      .def_readwrite("autotuneModelSize", &fasttext::Args::autotuneModelSize)
      .def("setManual", [](fasttext::Args &m, const std::string &argName)
           { m.setManual(argName); });

  py::enum_<fasttext::model_name>(m, "model_name")
      .value("cbow", fasttext::model_name::cbow)
      .value("skipgram", fasttext::model_name::sg)
      .value("supervised", fasttext::model_name::sup)
      .export_values();

  py::enum_<fasttext::loss_name>(m, "loss_name")
      .value("hs", fasttext::loss_name::hs)
      .value("ns", fasttext::loss_name::ns)
      .value("softmax", fasttext::loss_name::softmax)
      .value("ova", fasttext::loss_name::ova)
      .export_values();

  py::enum_<fasttext::metric_name>(m, "metric_name")
      .value("f1score", fasttext::metric_name::f1score)
      .value("f1scoreLabel", fasttext::metric_name::f1scoreLabel)
      .value("precisionAtRecall", fasttext::metric_name::precisionAtRecall)
      .value(
          "precisionAtRecallLabel",
          fasttext::metric_name::precisionAtRecallLabel)
      .value("recallAtPrecision", fasttext::metric_name::recallAtPrecision)
      .value(
          "recallAtPrecisionLabel",
          fasttext::metric_name::recallAtPrecisionLabel)
      .export_values();

  m.def(
      "train",
      [](FastTextHandle &handle, fasttext::Args &a)
      {
        // GIL already released by call_guard, so not writeLock().
        std::unique_lock<std::shared_mutex> lock(handle.mu);
        if (a.hasAutotune())
        {
          fasttext::Autotune autotune(std::shared_ptr<fasttext::FastText>(
              &handle.ft, [](fasttext::FastText *) {}));
          autotune.train(a);
        }
        else
        {
          handle.ft.train(a);
        }
      },
      py::call_guard<py::gil_scoped_release>());

  py::class_<fasttext::Vector>(m, "Vector", py::buffer_protocol())
      .def(py::init<ssize_t>())
      .def_buffer([](fasttext::Vector &m) -> py::buffer_info
                  { return py::buffer_info(
                        m.data(),
                        sizeof(fasttext::real),
                        py::format_descriptor<fasttext::real>::format(),
                        1,
                        {m.size()},
                        {sizeof(fasttext::real)}); });

  py::class_<fasttext::DenseMatrix, std::shared_ptr<fasttext::DenseMatrix>>(
      m, "DenseMatrix", py::buffer_protocol(), py::module_local())
      .def(py::init<>())
      .def(py::init<ssize_t, ssize_t>())
      .def_buffer([](fasttext::DenseMatrix &m) -> py::buffer_info
                  { return py::buffer_info(
                        m.data(),
                        sizeof(fasttext::real),
                        py::format_descriptor<fasttext::real>::format(),
                        2,
                        {m.size(0), m.size(1)},
                        {sizeof(fasttext::real) * m.size(1),
                         sizeof(fasttext::real) * (int64_t)1}); });

  py::class_<fasttext::Meter>(m, "Meter")
      .def(py::init<bool>())
      .def("scoreVsTrue", &fasttext::Meter::scoreVsTrue)
      .def(
          "precisionRecallCurveLabel",
          (std::vector<std::pair<double, double>> (fasttext::Meter::*)(int32_t)
               const) &
              fasttext::Meter::precisionRecallCurve)
      .def(
          "precisionRecallCurve",
          (std::vector<std::pair<double, double>> (fasttext::Meter::*)() const) &
              fasttext::Meter::precisionRecallCurve)
      .def(
          "precisionAtRecallLabel",
          (double (fasttext::Meter::*)(int32_t, double) const) &
              fasttext::Meter::precisionAtRecall)
      .def(
          "precisionAtRecall",
          (double (fasttext::Meter::*)(double) const) &
              fasttext::Meter::precisionAtRecall)
      .def(
          "recallAtPrecisionLabel",
          (double (fasttext::Meter::*)(int32_t, double) const) &
              fasttext::Meter::recallAtPrecision)
      .def(
          "recallAtPrecision",
          (double (fasttext::Meter::*)(double) const) &
              fasttext::Meter::recallAtPrecision);

  py::class_<FastTextHandle>(m, "fasttext")
      .def(py::init<>())
      .def(
          "getArgs",
          [](FastTextHandle &w)
          {
            auto lock = readLock(w);
            return w.ft.getArgs();
          })
      .def(
          "getInputMatrix",
          [](FastTextHandle &w)
          {
            auto lock = readLock(w);
            return w.ft.getInputMatrix();
          })
      .def(
          "getOutputMatrix",
          [](FastTextHandle &w)
          {
            auto lock = readLock(w);
            return w.ft.getOutputMatrix();
          })
      .def(
          "setMatrices",
          [](FastTextHandle &w,
             py::buffer inputMatrixBuffer,
             py::buffer outputMatrixBuffer)
          {
            py::buffer_info inputMatrixInfo = inputMatrixBuffer.request();
            py::buffer_info outputMatrixInfo = outputMatrixBuffer.request();

            auto lock = writeLock(w);
            w.ft.setMatrices(
                std::make_shared<fasttext::DenseMatrix>(
                    inputMatrixInfo.shape[0],
                    inputMatrixInfo.shape[1],
                    static_cast<float *>(inputMatrixInfo.ptr)),
                std::make_shared<fasttext::DenseMatrix>(
                    outputMatrixInfo.shape[0],
                    outputMatrixInfo.shape[1],
                    static_cast<float *>(outputMatrixInfo.ptr)));
          })
      .def(
          "loadModel",
          [](FastTextHandle &w, std::string s)
          {
            auto lock = writeLock(w);
            w.ft.loadModel(s);
          })
      .def(
          "saveModel",
          [](FastTextHandle &w, std::string s)
          {
            auto lock = readLock(w);
            w.ft.saveModel(s);
          })
      .def(
          "test",
          [](FastTextHandle &w,
             const std::string &filename,
             int32_t k,
             fasttext::real threshold)
          {
            std::ifstream ifs(filename);
            if (!ifs.is_open())
            {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(false);
            auto lock = readLock(w);
            w.ft.test(ifs, k, threshold, meter);
            ifs.close();
            return std::tuple<int64_t, double, double>(
                meter.nexamples(), meter.precision(), meter.recall());
          })
      .def(
          "getMeter",
          [](FastTextHandle &w, const std::string &filename, int32_t k)
          {
            std::ifstream ifs(filename);
            if (!ifs.is_open())
            {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(true);
            auto lock = readLock(w);
            w.ft.test(ifs, k, 0.0, meter);
            ifs.close();

            return meter;
          })
      .def(
          "getSentenceVector",
          [](FastTextHandle &w,
             fasttext::Vector &v,
             const std::string text)
          {
            std::stringstream ioss(text);
            auto lock = readLock(w);
            w.ft.getSentenceVector(ioss, v);
          })
      .def(
          "tokenize",
          [](FastTextHandle &w, const std::string text)
          {
            auto lock = readLock(w);
            std::vector<std::string> text_split;
            std::shared_ptr<const fasttext::Dictionary> d =
                w.ft.getDictionary();
            std::stringstream ioss(text);
            std::string token;
            while (!ioss.eof())
            {
              while (d->readWord(ioss, token))
              {
                text_split.push_back(token);
              }
            }
            return text_split;
          })
      .def(
          "getLine",
          [](FastTextHandle &w,
             const std::string text,
             const char *onUnicodeError)
          {
            std::pair<std::vector<std::string>, std::vector<std::string>> tokens;
            {
              auto lock = readLock(w);
              tokens = getLineTokens(w.ft, text);
            }
            return std::make_pair(
                castToPythonString(tokens.first, onUnicodeError),
                castToPythonString(tokens.second, onUnicodeError));
          })
      .def(
          "multilineGetLine",
          [](FastTextHandle &w,
             const std::vector<std::string> lines,
             const char *onUnicodeError)
          {
            std::vector<
                std::pair<std::vector<std::string>, std::vector<std::string>>>
                all_tokens;
            {
              auto lock = readLock(w);
              for (const auto &text : lines)
              {
                all_tokens.push_back(getLineTokens(w.ft, text));
              }
            }
            std::vector<std::vector<py::str>> all_words;
            std::vector<std::vector<py::str>> all_labels;
            for (const auto &tokens : all_tokens)
            {
              all_words.push_back(castToPythonString(tokens.first, onUnicodeError));
              all_labels.push_back(
                  castToPythonString(tokens.second, onUnicodeError));
            }
            return std::pair<
                std::vector<std::vector<py::str>>,
                std::vector<std::vector<py::str>>>(all_words, all_labels);
          })
      .def(
          "getVocab",
          [](FastTextHandle &w, const char *onUnicodeError)
          {
            std::vector<std::string> vocab;
            std::vector<int64_t> vocab_freq;
            {
              auto lock = readLock(w);
              std::shared_ptr<const fasttext::Dictionary> d =
                  w.ft.getDictionary();
              vocab_freq = d->getCounts(fasttext::entry_type::word);
              for (size_t i = 0; i < vocab_freq.size(); i++)
              {
                vocab.push_back(d->getWord(i));
              }
            }
            return std::pair<std::vector<py::str>, std::vector<int64_t>>(
                castToPythonString(vocab, onUnicodeError), vocab_freq);
          })
      .def(
          "getLabels",
          [](FastTextHandle &w, const char *onUnicodeError)
          {
            std::vector<std::string> labels;
            std::vector<int64_t> labels_freq;
            {
              auto lock = readLock(w);
              std::shared_ptr<const fasttext::Dictionary> d =
                  w.ft.getDictionary();
              labels_freq = d->getCounts(fasttext::entry_type::label);
              for (size_t i = 0; i < labels_freq.size(); i++)
              {
                labels.push_back(d->getLabel(i));
              }
            }
            return std::pair<std::vector<py::str>, std::vector<int64_t>>(
                castToPythonString(labels, onUnicodeError), labels_freq);
          })
      .def(
          "quantize",
          [](FastTextHandle &w,
             const std::string input,
             bool qout,
             int32_t cutoff,
             bool retrain,
             int epoch,
             double lr,
             int thread,
             int verbose,
             int32_t dsub,
             bool qnorm)
          {
            fasttext::Args qa = fasttext::Args();
            qa.input = input;
            qa.qout = qout;
            qa.cutoff = cutoff;
            qa.retrain = retrain;
            qa.epoch = epoch;
            qa.lr = lr;
            qa.thread = thread;
            qa.verbose = verbose;
            qa.dsub = dsub;
            qa.qnorm = qnorm;
            auto lock = writeLock(w);
            w.ft.quantize(qa);
          })
      .def(
          "predict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](FastTextHandle &w,
             const std::string text,
             int32_t k,
             fasttext::real threshold,
             const char *onUnicodeError)
          {
            std::stringstream ioss(text);
            std::vector<std::pair<fasttext::real, std::string>> predictions;
            {
              auto lock = readLock(w);
              w.ft.predictLine(ioss, predictions, k, threshold);
            }
            return castToPythonString(predictions, onUnicodeError);
          })
      .def(
          "multilinePredict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](FastTextHandle &w,
             const std::vector<std::string> &lines,
             int32_t k,
             fasttext::real threshold,
             const char *onUnicodeError)
          {
            std::vector<std::vector<std::pair<fasttext::real, std::string>>>
                allPredictions(lines.size());
            {
              auto lock = readLock(w);
              for (size_t i = 0; i < lines.size(); i++)
              {
                std::stringstream ioss(lines[i]);
                w.ft.predictLine(ioss, allPredictions[i], k, threshold);
              }
            }

            std::vector<py::array_t<fasttext::real>> allProbabilities;
            std::vector<std::vector<py::str>> allLabels;
            for (const auto &predictions : allPredictions)
            {
              std::vector<fasttext::real> probabilities;
              std::vector<py::str> labels;

              for (const auto &prediction : predictions)
              {
                probabilities.push_back(prediction.first);
                labels.push_back(
                    castToPythonString(prediction.second, onUnicodeError));
              }

              allProbabilities.emplace_back(
                  probabilities.size(), probabilities.data());
              allLabels.push_back(labels);
            }

            return make_pair(allLabels, allProbabilities);
          })
      .def(
          "testLabel",
          [](FastTextHandle &w,
             const std::string filename,
             int32_t k,
             fasttext::real threshold)
          {
            std::ifstream ifs(filename);
            if (!ifs.is_open())
            {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(false);
            std::vector<std::string> labels;
            {
              auto lock = readLock(w);
              w.ft.test(ifs, k, threshold, meter);
              std::shared_ptr<const fasttext::Dictionary> d =
                  w.ft.getDictionary();
              for (int32_t i = 0; i < d->nlabels(); i++)
              {
                labels.push_back(d->getLabel(i));
              }
            }
            std::unordered_map<std::string, py::dict> returnedValue;
            for (int32_t i = 0; i < static_cast<int32_t>(labels.size()); i++)
            {
              returnedValue[labels[i]] = py::dict(
                  "precision"_a = meter.precision(i),
                  "recall"_a = meter.recall(i),
                  "f1score"_a = meter.f1Score(i));
            }

            return returnedValue;
          })
      .def(
          "getWordId",
          [](FastTextHandle &w, const std::string &word)
          {
            auto lock = readLock(w);
            return w.ft.getWordId(word);
          })
      .def(
          "getSubwordId",
          [](FastTextHandle &w, const std::string word)
          {
            auto lock = readLock(w);
            return w.ft.getSubwordId(word);
          })
      .def(
          "getLabelId",
          [](FastTextHandle &w, const std::string &label)
          {
            auto lock = readLock(w);
            return w.ft.getLabelId(label);
          })
      .def(
          "getInputVector",
          [](FastTextHandle &w, fasttext::Vector &vec, int32_t ind)
          {
            auto lock = readLock(w);
            w.ft.getInputVector(vec, ind);
          })
      .def(
          "getWordVector",
          [](FastTextHandle &w,
             fasttext::Vector &vec,
             const std::string word)
          {
            auto lock = readLock(w);
            w.ft.getWordVector(vec, word);
          })
      .def(
          "getNN",
          [](FastTextHandle &w,
             const std::string &word,
             int32_t k,
             const char *onUnicodeError)
          {
            std::vector<std::pair<fasttext::real, std::string>> nn;
            {
              auto lock = readLock(w);
              nn = w.ft.getNN(word, k);
            }
            return castToPythonString(nn, onUnicodeError);
          })
      .def(
          "getAnalogies",
          [](FastTextHandle &w,
             const std::string &wordA,
             const std::string &wordB,
             const std::string &wordC,
             int32_t k,
             const char *onUnicodeError)
          {
            std::vector<std::pair<fasttext::real, std::string>> analogies;
            {
              auto lock = readLock(w);
              analogies = w.ft.getAnalogies(k, wordA, wordB, wordC);
            }
            return castToPythonString(analogies, onUnicodeError);
          })
      .def(
          "getSubwords",
          [](FastTextHandle &w,
             const std::string word,
             const char *onUnicodeError)
          {
            std::vector<std::string> subwords;
            std::vector<int32_t> ngrams;
            {
              auto lock = readLock(w);
              std::shared_ptr<const fasttext::Dictionary> d =
                  w.ft.getDictionary();
              d->getSubwords(word, ngrams, subwords);
            }
            return std::pair<std::vector<py::str>, std::vector<int32_t>>(
                castToPythonString(subwords, onUnicodeError), ngrams);
          })
      .def(
          "isQuant",
          [](FastTextHandle &w)
          {
            auto lock = readLock(w);
            return w.ft.isQuant();
          });
}
