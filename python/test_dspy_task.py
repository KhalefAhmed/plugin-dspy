from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/main/resources/scripts/dspy_task.py"
spec = importlib.util.spec_from_file_location("dspy_task", SCRIPT)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
ANSWER = "billing"
CUSTOM_PROGRAM = """
import dspy

class Translator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("sentence -> translation")

    def forward(self, sentence):
        return self.translate(sentence=sentence)

def build_program():
    return Translator()

def translation_metric(example, prediction, trace=None):
    return 0.75 if prediction.translation == "translated" else 0.0
"""


class FakeLLM(BaseHTTPRequestHandler):
    models = []
    statuses = []
    formats = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.models.append(body["model"])
        self.formats.append("JSON" if "response_format" in body else "CHAT")
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            payload = json.dumps({"error": {"message": "provider-private-payload", "type": "api_error"}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        system = body["messages"][0]["content"]
        fields = {"translation": "translated"} if "translation" in system else {"category": ANSWER}
        if "entities" in system:
            fields = {"entities": '["Alice"]'}
        if "confidence" in system:
            fields["confidence"] = "0.9"
        if "reasoning" in system:
            fields = {"reasoning": "This is an invoice request.", **fields}
        content = "\n\n".join(f"[[ ## {field} ## ]]\n{value}" for field, value in fields.items())
        if "response_format" in body:
            content = json.dumps(fields)
        payload = json.dumps({
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content if "response_format" in body else content + "\n\n[[ ## completed ## ]]",
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


def record(identifier="one", reference=True):
    value = {
        "schemaVersion": 1,
        "id": identifier,
        "inputs": {"text": "Please send me my invoice."},
    }
    if reference:
        value["reference"] = {"category": ANSWER}
    return value


def dataset(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


class DspyTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        FakeLLM.models.clear()
        FakeLLM.statuses.clear()
        FakeLLM.formats.clear()

    def config(self, operation, **overrides):
        return {
            "schemaVersion": 1,
            "operation": operation,
            "maxLlmCalls": 10,
            "maxRetries": 2,
            "requestTimeoutSeconds": 5,
            "maxTokens": 256,
            "studentModel": "openai/student",
            "teacherModel": "openai/teacher",
            "optimizer": "BOOTSTRAP_FEW_SHOT",
            "seed": 42,
            "maxBootstrappedDemos": 1,
            "maxLabeledDemos": 1,
            "metricThreshold": 1.0,
            "signature": "text -> category",
            "predictor": "PREDICT",
            "programFactory": "build_program",
            **overrides,
        }

    def run_task(self, directory, config):
        config = dict(config)
        files = {
            "program": "user_program.py", "train": "train.jsonl", "validation": "validation.jsonl",
            "dataset": "dataset.jsonl", "artifact": "artifact.json", "candidate": "candidate.json",
            "result": "optimization-result.json", "predictions": "predictions.jsonl", "summary": "prediction-summary.json",
            **config.get("files", {}),
        }
        if not (directory / files["program"]).is_file():
            files.pop("program")
        if not (directory / files["artifact"]).is_file():
            files.pop("artifact")
            if config.get("signature") is not None:
                config["program"] = {
                    "kind": "declarative",
                    "signature": config["signature"],
                    "predictor": config["predictor"],
                    "instructions": config.get("instructions"),
                }
            else:
                config["program"] = {
                    "kind": "python",
                    "factory": config["programFactory"],
                    "sha256": task.digest(directory / files["program"]),
                }
        config["files"] = files
        if config["operation"] == "optimize" and not config.get("metricFunction"):
            with (directory / files["train"]).open(encoding="utf-8") as source:
                config["referenceFields"] = list(json.loads(source.readline())["reference"])
        if config.get("evaluate"):
            with (directory / files["dataset"]).open(encoding="utf-8") as source:
                config["referenceFields"] = list(json.loads(source.readline())["reference"])
        task.write_json(directory / "config.json", config)
        endpoint = f"http://127.0.0.1:{self.server.server_port}/v1"
        environment = {
            **os.environ,
            "DSPY_STUDENT_API_BASE": endpoint,
            "DSPY_TEACHER_API_BASE": endpoint,
            "DSPY_STUDENT_API_KEY": "test-student-secret",
            "DSPY_TEACHER_API_KEY": "test-teacher-secret",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        return subprocess.run(
            [sys.executable, str(SCRIPT), "config.json"],
            cwd=directory, env=environment, capture_output=True, text=True, timeout=60,
        )

    def prepare_optimization(self):
        dataset(self.directory / "train.jsonl", [record("train")])
        dataset(self.directory / "validation.jsonl", [record("validation")])

    def test_round_trip_in_separate_processes_uses_teacher_and_preserves_inputs(self):
        self.prepare_optimization()
        result = self.run_task(self.directory, self.config("optimize"))
        self.assertEqual(result.returncode, 0, result.stderr)
        artifact = json.loads((self.directory / "candidate.json").read_text())
        result_data = json.loads((self.directory / "optimization-result.json").read_text())
        self.assertEqual(FakeLLM.models, ["teacher", "student"])
        self.assertEqual(result_data["validationScore"], 1)
        self.assertIsNone(artifact["state"]["predictor"]["lm"])
        self.assertNotIn("test-student-secret", json.dumps(artifact))
        self.assertNotIn("test-teacher-secret", json.dumps(artifact))

        with tempfile.TemporaryDirectory() as prediction_directory:
            target = Path(prediction_directory)
            (target / "artifact.json").write_bytes((self.directory / "candidate.json").read_bytes())
            question = record("test", reference=False)
            question["inputs"]["text"] += " {{ this is data, not a template }}"
            dataset(target / "dataset.jsonl", [question])
            config = self.config("predict")
            config.pop("signature")
            result = self.run_task(target, config)
            self.assertEqual(result.returncode, 0, result.stderr)
            prediction = json.loads((target / "predictions.jsonl").read_text())
            self.assertEqual(prediction["id"], "test")
            self.assertEqual(prediction["inputs"], question["inputs"])
            self.assertEqual(prediction["outputs"]["category"], ANSWER)
            self.assertEqual(prediction["artifactVersion"], artifact["artifactVersion"])
            self.assertEqual(FakeLLM.models, ["teacher", "student", "student"])

    def test_budget_fails_without_publishing_artifacts(self):
        self.prepare_optimization()
        result = self.run_task(self.directory, self.config("optimize", maxLlmCalls=1))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("budget exceeded", result.stderr)
        self.assertEqual(FakeLLM.models, ["teacher"])
        self.assertFalse((self.directory / "candidate.json").exists())

    def test_rejects_tampered_artifacts_and_missing_answers(self):
        self.prepare_optimization()
        result = self.run_task(self.directory, self.config("optimize"))
        self.assertEqual(result.returncode, 0, result.stderr)
        artifact = json.loads((self.directory / "candidate.json").read_text())
        artifact["studentModel"] = "openai/other"
        task.write_json(self.directory / "artifact.json", artifact)
        dataset(self.directory / "dataset.jsonl", [record("test", reference=False)])
        before = list(FakeLLM.models)
        result = self.run_task(self.directory, self.config("predict"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("integrity", result.stderr)
        self.assertEqual(FakeLLM.models, before)
        self.assertFalse((self.directory / "predictions.jsonl").exists())
        metric = task.make_metric({"referenceFields": ["category"]}, None)
        example = task.dspy.Example(category=ANSWER)
        prediction = task.dspy.Prediction(other="")
        with self.assertRaisesRegex(ValueError, "reference fields"):
            metric(example, prediction)

    def test_default_metric_allows_extra_outputs_and_rejects_missing_fields(self):
        metric = task.make_metric({"referenceFields": ["category", "confidence"]}, None)
        example = task.dspy.Example(category=ANSWER, confidence=0.9)
        prediction = task.dspy.Prediction(category=ANSWER, confidence=0.9, reasoning="Explanation")
        self.assertEqual(metric(example, prediction), 1.0)
        incomplete = task.dspy.Prediction(category=ANSWER, reasoning="Explanation")
        with self.assertRaisesRegex(ValueError, "reference fields"):
            metric(example, incomplete)

    def test_custom_metric_receives_bootstrap_trace(self):
        from types import SimpleNamespace

        seen = []

        def score(example, prediction, trace):
            seen.append(trace)
            return 0.75

        metric = task.make_metric({"metricFunction": "score"}, SimpleNamespace(score=score))
        trace = [("predictor", {"source": "bonjour"}, {"translation": "hello"})]
        self.assertEqual(metric(None, None, trace), 0.75)
        self.assertIs(seen[0], trace)

    def test_direct_prediction_and_chain_of_thought(self):
        dataset(self.directory / "dataset.jsonl", [record("test", reference=False)])
        cases = [
            ("text -> category", "CHAIN_OF_THOUGHT", {"category": ANSWER, "reasoning": "This is an invoice request."}),
            ("text -> entities: list[str]", "PREDICT", {"entities": ["Alice"]}),
            ("text -> category, confidence: float", "PREDICT", {"category": ANSWER, "confidence": 0.9}),
        ]
        for signature, predictor, expected in cases:
            with self.subTest(signature=signature):
                config = self.config("predict", model="openai/student", predictor=predictor, signature=signature)
                result = self.run_task(self.directory, config)
                self.assertEqual(result.returncode, 0, result.stderr)
                prediction = json.loads((self.directory / "predictions.jsonl").read_text())
                self.assertEqual(prediction["outputs"], expected)

    def test_prediction_preflights_late_invalid_rows_and_publishes_only_complete_batches(self):
        rows = [record(str(index), reference=False) for index in range(3)]
        rows[0]["inputs"]["text"] = "{{ this must remain literal }}"
        invalid = record("invalid", reference=False)
        invalid["inputs"] = {"wrong_field": "invoice"}
        dataset(self.directory / "dataset.jsonl", rows + [invalid])
        config = self.config("predict", model="openai/student", maxLlmCalls=2)
        result = self.run_task(self.directory, config)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(FakeLLM.models, [])
        dataset(self.directory / "dataset.jsonl", rows)
        result = self.run_task(self.directory, config)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(FakeLLM.models, ["student", "student"])
        self.assertFalse((self.directory / "predictions.jsonl").exists())
        result = self.run_task(self.directory, {**config, "maxLlmCalls": 3})
        self.assertEqual(result.returncode, 0, result.stderr)
        predictions = [json.loads(line) for line in (self.directory / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual([prediction["inputs"] for prediction in predictions], [row["inputs"] for row in rows])

    def test_custom_program_metric_and_source_hash(self):
        (self.directory / "user_program.py").write_text(CUSTOM_PROGRAM)
        for path, identifier in (("train.jsonl", "train"), ("validation.jsonl", "validation")):
            dataset(self.directory / path, [{
                "schemaVersion": 1, "id": identifier, "inputs": {"sentence": "bonjour"},
            }])
        config = self.config("optimize", metricFunction="translation_metric", metricThreshold=0.5)
        config.pop("signature")
        result = self.run_task(self.directory, config)
        self.assertEqual(result.returncode, 0, result.stderr)
        result_data = json.loads((self.directory / "optimization-result.json").read_text())
        artifact = json.loads((self.directory / "candidate.json").read_text())
        self.assertEqual(result_data["validationScore"], 0.75)
        self.assertEqual(artifact["program"]["kind"], "python")
        with tempfile.TemporaryDirectory() as prediction_directory:
            target = Path(prediction_directory)
            (target / "user_program.py").write_text(CUSTOM_PROGRAM)
            (target / "artifact.json").write_bytes((self.directory / "candidate.json").read_bytes())
            dataset(target / "dataset.jsonl", [{
                "schemaVersion": 1, "id": "test", "inputs": {"sentence": "salut"},
            }])
            result = self.run_task(target, self.config("predict", signature=None))
            self.assertEqual(result.returncode, 0, result.stderr)
            prediction = json.loads((target / "predictions.jsonl").read_text())
            self.assertEqual(prediction["outputs"], {"translation": "translated"})
            (target / "predictions.jsonl").unlink()
            (target / "user_program.py").write_text(CUSTOM_PROGRAM + "\n# different source\n")
            result = self.run_task(target, self.config("predict", signature=None))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source hash mismatch", result.stderr)
            self.assertFalse((target / "predictions.jsonl").exists())

    def test_custom_program_round_trip_with_configured_nested_paths(self):
        files = {
            "program": "modules/my translator.py",
            "train": "inputs/examples.jsonl",
            "validation": "inputs/held out.jsonl",
            "candidate": "compiled/model.json",
            "result": "reports/evaluation.json",
            "artifact": "saved/original.json",
            "dataset": "requests/batch.jsonl",
            "predictions": "answers/output.jsonl",
            "summary": "reports/prediction.json",
        }
        source = self.directory / files["program"]
        source.parent.mkdir()
        source.write_text(CUSTOM_PROGRAM)
        for role, identifier in (("train", "train"), ("validation", "validation")):
            dataset(self.directory / files[role], [{
                "schemaVersion": 1, "id": identifier, "inputs": {"sentence": "bonjour"},
            }])
        config = self.config("optimize", signature=None, metricFunction="translation_metric", files=files)
        result = self.run_task(self.directory, config)
        self.assertEqual(result.returncode, 0, result.stderr)
        artifact = self.directory / files["artifact"]
        artifact.parent.mkdir()
        artifact.write_bytes((self.directory / files["candidate"]).read_bytes())
        dataset(self.directory / files["dataset"], [{
            "schemaVersion": 1, "id": "test", "inputs": {"sentence": "salut"},
        }])
        result = self.run_task(self.directory, self.config("predict", signature=None, files=files))
        self.assertEqual(result.returncode, 0, result.stderr)
        prediction = json.loads((self.directory / files["predictions"]).read_text())
        self.assertEqual(prediction["outputs"], {"translation": "translated"})
        self.assertTrue((self.directory / files["summary"]).is_file())
        self.assertFalse((self.directory / "candidate.json").exists())
        self.assertFalse((self.directory / "predictions.jsonl").exists())

    def test_transient_retries_consume_budget_and_redact_provider_payloads(self):
        dataset(self.directory / "dataset.jsonl", [record("test", reference=False)])
        FakeLLM.statuses.extend([429, 503])
        result = self.run_task(self.directory, self.config("predict", model="openai/student", maxLlmCalls=3))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.directory / "prediction-summary.json").read_text())
        self.assertEqual(summary["modelCalls"], {"openai/student": 3})
        self.assertNotIn("provider-private-payload", result.stderr)

        (self.directory / "predictions.jsonl").unlink()
        FakeLLM.models.clear()
        FakeLLM.statuses.append(429)
        result = self.run_task(self.directory, self.config("predict", model="openai/student", maxLlmCalls=1))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("budget exceeded", result.stderr)
        self.assertEqual(FakeLLM.models, ["student"])
        self.assertFalse((self.directory / "predictions.jsonl").exists())

    def test_non_retryable_errors_and_disabled_or_exhausted_retries(self):
        dataset(self.directory / "dataset.jsonl", [record("test", reference=False)])
        for status, retries, attempts in ((401, 2, 1), (400, 2, 1), (429, 0, 1), (503, 1, 2)):
            with self.subTest(status=status, retries=retries):
                FakeLLM.models.clear()
                FakeLLM.statuses[:] = [status] * attempts
                result = self.run_task(self.directory, self.config("predict", model="openai/student", maxRetries=retries))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(FakeLLM.models), attempts)
                self.assertIn(f"HTTP {status}", result.stderr)
                self.assertNotIn("provider-private-payload", result.stderr)
                self.assertFalse((self.directory / "predictions.jsonl").exists())

    def test_timeout_retries_and_concurrent_budget_limits(self):
        from concurrent.futures import ThreadPoolExecutor

        budget = task.Budget(3)
        task.BudgetLM.budget = budget
        lm = task.BudgetLM("openai/student", max_retries=1)
        error = task.dspy.LMTimeoutError("provider-private-payload", model="openai/student")
        with patch.object(task.dspy.LM, "__call__", side_effect=[error, ["ok"]]), patch.object(task.time, "sleep"):
            self.assertEqual(lm("invoice"), ["ok"])
        self.assertEqual(budget.counts, {"openai/student": 2})

        def consume(_):
            try:
                budget.consume("openai/student")
                return True
            except task.BudgetExceeded:
                return False

        with ThreadPoolExecutor(max_workers=8) as executor:
            self.assertEqual(sum(executor.map(consume, range(30))), 1)
        self.assertEqual(budget.counts, {"openai/student": 3})

    def test_evaluation_scores_all_rows_without_changing_prediction_outputs(self):
        rows = [record("correct"), record("incorrect")]
        rows[1]["reference"]["category"] = "technical"
        dataset(self.directory / "dataset.jsonl", rows)
        result = self.run_task(self.directory, self.config("predict", model="openai/student", evaluate=True))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.directory / "prediction-summary.json").read_text())
        self.assertEqual(summary["evaluationScore"], 0.5)
        self.assertEqual(summary["count"], 2)

    def test_prediction_does_not_retain_inputs_in_dspy_traces(self):
        dataset(self.directory / "dataset.jsonl", [record(str(index), reference=False) for index in range(20)])
        config = self.config("predict", model="openai/student")
        config["program"] = {"kind": "declarative", "signature": "text -> category", "predictor": "PREDICT"}
        config["files"] = {
            "dataset": str(self.directory / "dataset.jsonl"),
            "predictions": str(self.directory / "predictions.jsonl"),
            "summary": str(self.directory / "summary.json"),
        }
        task.BudgetLM.budget = task.Budget(20)
        trace = []

        def completion(*args, **kwargs):
            return ["[[ ## category ## ]]\nbilling\n\n[[ ## completed ## ]]"]

        with task.dspy.context(trace=trace, disable_history=True), patch.object(task.dspy.LM, "__call__", new=completion):
            task.predict(config)
        self.assertEqual(trace, [])
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["count"], 20)

    def test_json_artifact_preserves_response_format_and_allows_explicit_override(self):
        self.prepare_optimization()
        result = self.run_task(self.directory, self.config("optimize", responseFormat="JSON"))
        self.assertEqual(result.returncode, 0, result.stderr)
        artifact = json.loads((self.directory / "candidate.json").read_text())
        self.assertEqual(artifact["responseFormat"], "JSON")
        self.assertEqual(FakeLLM.formats, ["JSON", "JSON"])
        task.write_json(self.directory / "artifact.json", artifact)
        dataset(self.directory / "dataset.jsonl", [record("test", reference=False)])
        config = self.config("predict", signature=None)
        result = self.run_task(self.directory, config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(FakeLLM.formats[-1], "JSON")
        summary = json.loads((self.directory / "prediction-summary.json").read_text())
        self.assertEqual(summary["responseFormat"], "JSON")
        result = self.run_task(self.directory, {**config, "responseFormat": "CHAT"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(FakeLLM.formats[-1], "CHAT")

        artifact.pop("responseFormat")
        artifact["artifactVersion"] = task.artifact_version(artifact)
        task.write_json(self.directory / "artifact.json", artifact)
        result = self.run_task(self.directory, config)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(FakeLLM.formats[-1], "CHAT")


if __name__ == "__main__":
    unittest.main()
