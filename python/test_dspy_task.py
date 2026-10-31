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

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.models.append(body["model"])
        system = body["messages"][0]["content"]
        fields = {"translation": "translated"} if "translation" in system else {"category": ANSWER}
        if "entities" in system:
            fields = {"entities": '["Alice"]'}
        if "confidence" in system:
            fields["confidence"] = "0.9"
        if "reasoning" in system:
            fields = {"reasoning": "This is an invoice request.", **fields}
        content = "\n\n".join(f"[[ ## {field} ## ]]\n{value}" for field, value in fields.items())
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
                    "content": content + "\n\n[[ ## completed ## ]]",
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

    def config(self, operation, **overrides):
        return {
            "schemaVersion": 1,
            "operation": operation,
            "maxLlmCalls": 10,
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
        if "datasets" not in config:
            roles = ("train", "validation") if config["operation"] == "optimize" else ("dataset",)
            config["datasets"] = {}
            for role in roles:
                with (directory / files[role]).open(encoding="utf-8") as source:
                    config["datasets"][role] = [json.loads(line) for line in source]
        if config["operation"] == "optimize" and not config.get("metricFunction"):
            config["referenceFields"] = list(config["datasets"]["train"][0]["reference"])
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

    def test_predict_uses_prepared_records_without_reading_dataset_files(self):
        row = record("prepared", reference=False)
        row["inputs"]["text"] = "{{ this must remain literal }}"
        config = self.config("predict", model="openai/student", datasets={"dataset": [row]})
        result = self.run_task(self.directory, config)
        self.assertEqual(result.returncode, 0, result.stderr)
        prediction = json.loads((self.directory / "predictions.jsonl").read_text())
        self.assertEqual(prediction["inputs"], row["inputs"])
        self.assertFalse((self.directory / "dataset.jsonl").exists())

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


if __name__ == "__main__":
    unittest.main()
