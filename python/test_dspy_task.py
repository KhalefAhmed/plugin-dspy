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
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(FakeLLM.models, ["teacher", "student"])
        self.assertEqual(manifest["validation"]["score"], 1)
        self.assertEqual(manifest["artifactSha256"], task.digest(self.directory / "candidate.json"))
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

    def test_rejects_invalid_datasets_before_model_calls(self):
        cases = [
            [],
            [record(), record()],
            [{**record(), "inputs": []}],
            [{key: value for key, value in record().items() if key != "reference"}],
            [{**record(), "schemaVersion": True}],
            [{**record(), "inputs": {"category": "overlaps with reference"}}],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                path = self.directory / "invalid.jsonl"
                dataset(path, rows)
                with self.assertRaises(ValueError):
                    task.read_dataset(path, supervised=True)
        self.assertEqual(FakeLLM.models, [])

    def test_overlap_and_budget_fail_without_publishing_artifacts(self):
        self.prepare_optimization()
        dataset(self.directory / "validation.jsonl", [record("train")])
        result = self.run_task(self.directory, self.config("optimize"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disjoint", result.stderr)
        self.assertEqual(FakeLLM.models, [])
        self.assertFalse((self.directory / "candidate.json").exists())

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
        metric = task.make_metric({}, None, [record()])
        with self.assertRaisesRegex(ValueError, "reference fields"):
            metric(task.dspy.Example(category=ANSWER), task.dspy.Prediction(other=""))

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
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(manifest["validation"]["score"], 0.75)
        self.assertEqual(manifest["program"]["kind"], "python")
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


if __name__ == "__main__":
    unittest.main()
