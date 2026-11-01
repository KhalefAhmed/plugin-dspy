import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys
import threading
import time
import unicodedata

import dspy
from pydantic import TypeAdapter


SCHEMA_VERSION = 1
DSPY_VERSION = "3.4.1"
OUTPUT_ADAPTER = TypeAdapter(dict[str, object])


class ContractError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    def __init__(self, error, retryable):
        status = error.status
        detail = f" (HTTP {status})" if type(status) is int else ""
        reason = "transient failure; retries exhausted" if retryable else "non-retryable failure"
        super().__init__(f"LLM request failed: {type(error).__name__}{detail}; {reason}. Provider payload redacted.")


class DeclarativeProgram(dspy.Module):
    def __init__(self, descriptor):
        super().__init__()
        signature = dspy.Signature(descriptor["signature"])
        if descriptor.get("instructions"):
            signature = signature.with_instructions(descriptor["instructions"])
        predictors = {"PREDICT": dspy.Predict, "CHAIN_OF_THOUGHT": dspy.ChainOfThought}
        if descriptor["predictor"] not in predictors:
            raise ContractError("Unsupported declarative predictor")
        self.predictor = predictors[descriptor["predictor"]](signature)

    def forward(self, **inputs):
        return self.predictor(**inputs)


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.counts = {}
        self.lock = threading.Lock()

    def consume(self, model):
        with self.lock:
            if sum(self.counts.values()) >= self.limit:
                raise BudgetExceeded("Maximum model-call budget exceeded")
            self.counts[model] = self.counts.get(model, 0) + 1


class BudgetLM(dspy.LM):
    # One task per process: copies made by BootstrapFewShot share this budget.
    budget = None

    def __init__(self, *args, max_retries=2, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries

    def __call__(self, *args, **kwargs):
        for attempt in range(self._max_retries + 1):
            self.budget.consume(self.model)
            try:
                return super().__call__(*args, **kwargs)
            except dspy.LMError as error:
                retryable = isinstance(error, (dspy.LMRateLimitError, dspy.LMTimeoutError, dspy.LMTransportError))
                retryable = retryable or error.status in (500, 502, 503, 504)
                if not retryable or attempt == self._max_retries:
                    raise ProviderError(error, retryable) from None
                print(
                    f"Transient LLM failure ({type(error).__name__}); retry {attempt + 1}/{self._max_retries}.",
                    file=sys.stderr,
                )
                time.sleep(min(2 ** attempt, 8))
        raise ContractError("maxRetries must not be negative")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical(value) + b"\n")


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def versioned(value, location):
    if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != SCHEMA_VERSION:
        raise ContractError(f"{location} requires schemaVersion: {SCHEMA_VERSION}")


def normalized(value):
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", " ".join(value.casefold().split()))
    return canonical(value)


def prediction_outputs(prediction):
    outputs = OUTPUT_ADAPTER.dump_python(dict(prediction.items()), mode="json")
    if not outputs:
        raise ContractError("The program returned no output fields")
    canonical(outputs)
    return outputs


def examples(records):
    return [
        dspy.Example(**row["inputs"], **row.get("reference", {})).with_inputs(*row["inputs"])
        for row in records
    ]


def read_dataset(path):
    count = 0
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ContractError(f"Invalid dataset JSON at line {line_number}") from None
            count += 1
            yield row
    if count == 0:
        raise ContractError("Dataset must not be empty")


def load_user_module(path):
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("user_program", path)
    if spec is None or spec.loader is None:
        raise ContractError("The configured program path must point to a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_program(descriptor, module) -> dspy.Module:
    if descriptor.get("kind") == "declarative":
        if module is not None:
            raise ContractError("A declarative artifact cannot be combined with programUri")
        return DeclarativeProgram(descriptor)
    if descriptor.get("kind") != "python":
        raise ContractError("Unknown program kind")
    if module is None or descriptor.get("sha256") != digest(module.__file__):
        raise ContractError("The custom artifact requires its original programUri (source hash mismatch)")
    factory = getattr(module, descriptor["factory"], None)
    if not callable(factory):
        raise ContractError("programFactory must name a callable in programUri")
    program = factory()
    if not isinstance(program, dspy.Module):
        raise ContractError("programFactory must return a dspy.Module")
    return program


def make_metric(config, module):
    name = config.get("metricFunction")
    if name:
        custom = getattr(module, name, None) if module is not None else None
        if not callable(custom):
            raise ContractError("metricFunction must name a callable in programUri")
    else:
        fields = set(config["referenceFields"])

        def custom(example, prediction, _trace=None):
            outputs = prediction_outputs(prediction)
            if fields.difference(outputs):
                raise ContractError("The program did not return all reference fields")
            return all(normalized(example[field]) == normalized(outputs[field]) for field in fields)

    def metric(example, prediction, trace=None):
        score = custom(example, prediction, trace)
        if not isinstance(score, (bool, int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ContractError("The metric must return a finite score between 0 and 1")
        return float(score)

    return metric


def validate_fields(records, descriptor):
    signature = dspy.Signature(descriptor["signature"]) if descriptor["kind"] == "declarative" else None
    for row in records:
        fields = row["inputs"] | row.get("reference", {})
        if any(not key.isidentifier() or key.startswith("_") for key in fields):
            raise ContractError("Dataset field names must be public Python identifiers")
        if signature is None:
            continue
        if set(row["inputs"]) != set(signature.input_fields):
            raise ContractError("Dataset inputs must match the declared signature fields")
        if row.get("reference") and set(row["reference"]) != set(signature.output_fields):
            raise ContractError("Dataset references must match the declared signature output fields")


def make_lm(model, role, config):
    options = {
        "cache": False,
        "num_retries": 0,
        "max_tokens": config["maxTokens"],
        "timeout": config["requestTimeoutSeconds"],
        "engine": "litellm",
    }
    for option, suffix in (("api_key", "API_KEY"), ("api_base", "API_BASE")):
        value = os.environ.get(f"DSPY_{role}_{suffix}")
        if value:
            options[option] = value
    return BudgetLM(text(model, f"{role} model"), max_retries=config["maxRetries"], **options)


def make_adapter(response_format):
    if response_format == "CHAT":
        return dspy.ChatAdapter(use_json_adapter_fallback=False)
    if response_format == "JSON":
        return dspy.JSONAdapter()
    raise ContractError("Unsupported responseFormat; expected CHAT or JSON")


def artifact_version(artifact):
    return hashlib.sha256(canonical({key: value for key, value in artifact.items() if key != "artifactVersion"})).hexdigest()


def make_artifact(program: dspy.Module, descriptor, model, response_format="CHAT"):
    program.set_lm(None)
    artifact = {
        "schemaVersion": SCHEMA_VERSION,
        "program": descriptor,
        "dspyVersion": DSPY_VERSION,
        "pythonVersion": platform.python_version(),
        "studentModel": model,
        "responseFormat": response_format,
        "state": program.dump_state(),
    }
    artifact["artifactVersion"] = artifact_version(artifact)
    return artifact


def validate_state(state):
    if not isinstance(state, dict):
        raise ContractError("Invalid program state")
    for value in state.values():
        if isinstance(value, dict):
            validate_state(value)
    if "signature" in state and "demos" in state and state.get("lm") is not None:
        raise ContractError("Artifacts must not contain a serialized LM")


def optimize(config):
    files = config["files"]
    train = list(read_dataset(files["train"]))
    validation = list(read_dataset(files["validation"]))
    descriptor = config["program"]
    validate_fields(train + validation, descriptor)
    module = load_user_module(files.get("program"))
    metric = make_metric(config, module)
    # Reproducible training order only; this RNG does not generate secrets or tokens.
    random.seed(config["seed"])
    random.shuffle(train)
    student_lm = make_lm(config["studentModel"], "STUDENT", config)
    teacher_lm = make_lm(config["teacherModel"], "TEACHER", config)
    student = build_program(descriptor, module)
    student.set_lm(student_lm)
    teacher = build_program(descriptor, module)
    teacher.set_lm(teacher_lm)
    response_format = config.get("responseFormat", "CHAT")
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        metric_threshold=config["metricThreshold"],
        teacher_settings={"lm": teacher_lm},
        max_bootstrapped_demos=config["maxBootstrappedDemos"],
        max_labeled_demos=config["maxLabeledDemos"],
        max_rounds=1,
        max_errors=1,
    )
    with dspy.context(lm=student_lm, adapter=make_adapter(response_format)):
        candidate = optimizer.compile(student, teacher=teacher, trainset=examples(train))
        candidate.set_lm(student_lm)
        scores = [
            metric(example, candidate(**example.inputs()))
            for example in examples(validation)
        ]
    artifact = make_artifact(candidate, descriptor, config["studentModel"], response_format)
    validate_state(artifact["state"])
    result = {
        "pythonVersion": platform.python_version(),
        "dspyVersion": DSPY_VERSION,
        "dependencyVersions": {
            name: importlib.metadata.version(name)
            for name in ("dspy", "litellm", "pydantic", "openai")
        },
        "validationScore": sum(scores) / len(scores),
        "modelCalls": dict(BudgetLM.budget.counts),
    }
    write_json(files["candidate"], artifact)
    write_json(files["result"], result)


def predict(config):
    files = config["files"]
    module = load_user_module(files.get("program"))
    if files.get("artifact") is not None:
        artifact = json.loads(Path(files["artifact"]).read_text(encoding="utf-8"))
        versioned(artifact, "artifact")
        if artifact.get("dspyVersion") != DSPY_VERSION:
            raise ContractError("Incompatible DSPy version")
        if artifact.get("artifactVersion") != artifact_version(artifact):
            raise ContractError("Artifact integrity check failed")
        descriptor = artifact["program"]
        validate_state(artifact["state"])
        program = build_program(descriptor, module)
        program.load_state(artifact["state"])
        model = text(config.get("model", artifact.get("studentModel")), "Student model")
        response_format = config.get("responseFormat", artifact.get("responseFormat", "CHAT"))
    else:
        descriptor = config["program"]
        program = build_program(descriptor, module)
        model = text(config.get("model"), "Student model")
        response_format = config.get("responseFormat", "CHAT")
        artifact = make_artifact(program, descriptor, model, response_format)
    # Preflight all rows before any billable request, without retaining the dataset.
    validate_fields(read_dataset(files["dataset"]), descriptor)
    metric = make_metric(config, None) if config.get("evaluate") else None
    lm = make_lm(model, "STUDENT", config)
    program.set_lm(lm)
    started = time.monotonic()
    target_path = Path(files["predictions"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial = target_path.with_name(target_path.name + ".partial")
    count = 0
    score = 0.0
    with dspy.context(lm=lm, adapter=make_adapter(response_format), trace=None), partial.open("wb") as target:
        for row in read_dataset(files["dataset"]):
            prediction = program(**row["inputs"])
            if metric is not None:
                score += metric(dspy.Example(**row["inputs"], **row["reference"]), prediction)
            target.write(canonical({
                "schemaVersion": SCHEMA_VERSION,
                "id": row["id"],
                "inputs": row["inputs"],
                "outputs": prediction_outputs(prediction),
                "artifactVersion": artifact["artifactVersion"],
            }) + b"\n")
            count += 1
    partial.replace(target_path)
    summary = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactVersion": artifact["artifactVersion"],
        "model": model,
        "responseFormat": response_format,
        "count": count,
        "durationSeconds": time.monotonic() - started,
        "modelCalls": dict(BudgetLM.budget.counts),
    }
    if metric is not None:
        summary["evaluationScore"] = score / count
    write_json(files["summary"], summary)


def report_error(error_type, error, _traceback):
    if isinstance(error, (ContractError, BudgetExceeded, ProviderError)):
        print(f"{error_type.__name__}: {error}", file=sys.stderr)
    else:
        print(f"DSPy execution failed ({error_type.__name__}). Provider details and program data are redacted.", file=sys.stderr)


def main(path):
    if importlib.metadata.version("dspy") != DSPY_VERSION:
        raise ContractError(f"This plugin requires dspy=={DSPY_VERSION}")
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    BudgetLM.budget = Budget(config["maxLlmCalls"])
    dspy.configure(adapter=make_adapter(config.get("responseFormat", "CHAT")), disable_history=True)
    logging.getLogger("dspy").setLevel(logging.ERROR)
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        logging.getLogger(name).disabled = True
    operation = config.get("operation")
    if operation == "optimize":
        optimize(config)
    elif operation == "predict":
        predict(config)
    else:
        raise ContractError("Unknown DSPy operation")


if __name__ == "__main__":
    sys.excepthook = report_error
    main(sys.argv[1])
