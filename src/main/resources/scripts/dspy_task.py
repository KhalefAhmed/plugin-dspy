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

    def consume(self, model):
        if sum(self.counts.values()) >= self.limit:
            raise BudgetExceeded("Maximum model-call budget exceeded")
        self.counts[model] = self.counts.get(model, 0) + 1


class BudgetLM(dspy.LM):
    # One task per process: copies made by BootstrapFewShot share this budget.
    budget = None

    def __call__(self, *args, **kwargs):
        self.budget.consume(self.model)
        return super().__call__(*args, **kwargs)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def versioned(value, location):
    if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != SCHEMA_VERSION:
        raise ContractError(f"{location} requires schemaVersion: {SCHEMA_VERSION}")


def read_dataset(path, supervised):
    records = []
    ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            location = f"{Path(path).name}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"Invalid JSON at {location}") from error
            versioned(record, location)
            identifier = text(record.get("id"), f"{location} id")
            if identifier in ids:
                raise ContractError(f"Duplicate id at {location}")
            ids.add(identifier)
            inputs = record.get("inputs")
            if not isinstance(inputs, dict):
                raise ContractError(f"{location} requires an inputs object")
            reference = record.get("reference", {})
            if not isinstance(reference, dict) or (supervised and not reference):
                raise ContractError(f"{location} requires a non-empty reference object")
            if set(inputs) & set(reference):
                raise ContractError(f"{location} input and reference fields must be disjoint")
            if any(not key.isidentifier() or key.startswith("_") for key in inputs | reference):
                raise ContractError(f"{location} field names must be public Python identifiers")
            canonical(record)
            records.append(record)
    if not records:
        raise ContractError(f"{Path(path).name} must not be empty")
    return records


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


def load_user_module():
    if not Path("user_program.py").is_file():
        return None
    spec = importlib.util.spec_from_file_location("user_program", "user_program.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def program_descriptor(config):
    if config.get("signature") is not None:
        if Path("user_program.py").is_file():
            raise ContractError("Provide either signature or programUri, not both")
        return {
            "kind": "declarative",
            "signature": text(config["signature"], "signature"),
            "predictor": config["predictor"],
            "instructions": config.get("instructions"),
        }
    if not Path("user_program.py").is_file():
        raise ContractError("Provide signature or programUri")
    return {
        "kind": "python",
        "factory": text(config["programFactory"], "programFactory"),
        "sha256": digest("user_program.py"),
    }


def build_program(descriptor, module):
    if descriptor.get("kind") == "declarative":
        if module is not None:
            raise ContractError("A declarative artifact cannot be combined with programUri")
        return DeclarativeProgram(descriptor)
    if descriptor.get("kind") != "python":
        raise ContractError("Unknown program kind")
    if module is None or descriptor.get("sha256") != digest("user_program.py"):
        raise ContractError("The custom artifact requires its original programUri (source hash mismatch)")
    factory = getattr(module, descriptor["factory"], None)
    if not callable(factory):
        raise ContractError("programFactory must name a callable in programUri")
    program = factory()
    if not isinstance(program, dspy.Module):
        raise ContractError("programFactory must return a dspy.Module")
    return program


def make_metric(config, module, records):
    name = config.get("metricFunction")
    if name:
        custom = getattr(module, name, None) if module is not None else None
        if not callable(custom):
            raise ContractError("metricFunction must name a callable in programUri")
    else:
        reference_fields = {frozenset(row["reference"]) for row in records}
        if len(reference_fields) != 1:
            raise ContractError("All reference objects must use the same output fields")
        fields = next(iter(reference_fields))

        def custom(example, prediction, trace=None):
            outputs = prediction_outputs(prediction)
            if not fields <= outputs.keys():
                raise ContractError("The program did not return all reference fields")
            return all(normalized(example[field]) == normalized(outputs[field]) for field in fields)

    def metric(example, prediction, trace=None):
        score = custom(example, prediction, trace)
        if not isinstance(score, (bool, int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ContractError("The metric must return a finite score between 0 and 1")
        return float(score)

    return metric


def validate_fields(records, descriptor):
    if descriptor["kind"] != "declarative":
        return
    signature = dspy.Signature(descriptor["signature"])
    inputs = set(signature.input_fields)
    outputs = set(signature.output_fields)
    for row in records:
        if set(row["inputs"]) != inputs:
            raise ContractError("Dataset inputs must match the declared signature fields")
        if row.get("reference") and set(row["reference"]) != outputs:
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
    return BudgetLM(text(model, f"{role} model"), **options)


def artifact_version(artifact):
    return hashlib.sha256(canonical({key: value for key, value in artifact.items() if key != "artifactVersion"})).hexdigest()


def make_artifact(program, descriptor, model):
    program.set_lm(None)
    artifact = {
        "schemaVersion": SCHEMA_VERSION,
        "program": descriptor,
        "dspyVersion": DSPY_VERSION,
        "pythonVersion": platform.python_version(),
        "studentModel": model,
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
    if config.get("optimizer") != "BOOTSTRAP_FEW_SHOT":
        raise ContractError("Unsupported optimizer")
    supervised = not config.get("metricFunction")
    train = read_dataset("train.jsonl", supervised=supervised)
    validation = read_dataset("validation.jsonl", supervised=supervised)
    if {row["id"] for row in train} & {row["id"] for row in validation}:
        raise ContractError("Training and validation IDs must be disjoint")
    descriptor = program_descriptor(config)
    validate_fields(train + validation, descriptor)
    module = load_user_module()
    metric = make_metric(config, module, train + validation)
    random.seed(config["seed"])
    random.shuffle(train)
    student_lm = make_lm(config["studentModel"], "STUDENT", config)
    teacher_lm = make_lm(config["teacherModel"], "TEACHER", config)
    student = build_program(descriptor, module)
    student.set_lm(student_lm)
    teacher = build_program(descriptor, module)
    teacher.set_lm(teacher_lm)
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        metric_threshold=config["metricThreshold"],
        teacher_settings={"lm": teacher_lm},
        max_bootstrapped_demos=config["maxBootstrappedDemos"],
        max_labeled_demos=config["maxLabeledDemos"],
        max_rounds=1,
        max_errors=1,
    )
    with dspy.context(lm=student_lm):
        candidate = optimizer.compile(student, teacher=teacher, trainset=examples(train))
        candidate.set_lm(student_lm)
        scores = [
            metric(example, candidate(**example.inputs()))
            for example in examples(validation)
        ]
    artifact = make_artifact(candidate, descriptor, config["studentModel"])
    validate_state(artifact["state"])
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactVersion": artifact["artifactVersion"],
        "artifactSha256": hashlib.sha256(canonical(artifact) + b"\n").hexdigest(),
        "program": descriptor,
        "pythonVersion": platform.python_version(),
        "dspyVersion": DSPY_VERSION,
        "studentModel": config["studentModel"],
        "teacherModel": config["teacherModel"],
        "optimizer": config["optimizer"],
        "seed": config["seed"],
        "maxBootstrappedDemos": config["maxBootstrappedDemos"],
        "maxLabeledDemos": config["maxLabeledDemos"],
        "metricThreshold": config["metricThreshold"],
        "maxLlmCalls": config["maxLlmCalls"],
        "maxTokens": config["maxTokens"],
        "requestTimeoutSeconds": config["requestTimeoutSeconds"],
        "dependencyVersions": {
            name: importlib.metadata.version(name)
            for name in ("dspy", "litellm", "pydantic", "openai")
        },
        "train": {"sha256": digest("train.jsonl"), "count": len(train)},
        "validation": {
            "sha256": digest("validation.jsonl"),
            "count": len(validation),
            "metric": config.get("metricFunction") or "normalized_exact_match_v1",
            "score": sum(scores) / len(scores),
        },
        "modelCalls": dict(BudgetLM.budget.counts),
    }
    write_json("candidate.json", artifact)
    write_json("manifest.json", manifest)


def predict(config):
    records = read_dataset("dataset.jsonl", supervised=False)
    module = load_user_module()
    if Path("artifact.json").is_file():
        artifact = json.loads(Path("artifact.json").read_text(encoding="utf-8"))
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
    else:
        descriptor = program_descriptor(config)
        program = build_program(descriptor, module)
        model = text(config.get("model"), "Student model")
        artifact = make_artifact(program, descriptor, model)
    validate_fields(records, descriptor)
    lm = make_lm(model, "STUDENT", config)
    program.set_lm(lm)
    started = time.monotonic()
    partial = Path("predictions.partial")
    with dspy.context(lm=lm), partial.open("wb") as target:
        for row in records:
            prediction = program(**row["inputs"])
            target.write(canonical({
                "schemaVersion": SCHEMA_VERSION,
                "id": row["id"],
                "inputs": row["inputs"],
                "outputs": prediction_outputs(prediction),
                "artifactVersion": artifact["artifactVersion"],
            }) + b"\n")
    partial.replace("predictions.jsonl")
    write_json("prediction-summary.json", {
        "schemaVersion": SCHEMA_VERSION,
        "artifactVersion": artifact["artifactVersion"],
        "model": model,
        "count": len(records),
        "durationSeconds": time.monotonic() - started,
        "modelCalls": dict(BudgetLM.budget.counts),
    })


def report_error(error_type, error, traceback):
    if isinstance(error, (ContractError, BudgetExceeded)):
        print(f"{error_type.__name__}: {error}", file=sys.stderr)
    else:
        print(f"DSPy execution failed ({error_type.__name__}). Provider details and program data are redacted.", file=sys.stderr)


def main(path):
    if importlib.metadata.version("dspy") != DSPY_VERSION:
        raise ContractError(f"This plugin requires dspy=={DSPY_VERSION}")
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    versioned(config, "configuration")
    for field in ("maxLlmCalls", "requestTimeoutSeconds", "maxTokens"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ContractError(f"{field} must be a positive integer")
    BudgetLM.budget = Budget(config["maxLlmCalls"])
    dspy.configure(adapter=dspy.ChatAdapter(use_json_adapter_fallback=False), disable_history=True)
    logging.getLogger("dspy").setLevel(logging.ERROR)
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
        logging.getLogger(name).disabled = True
    operation = config.get("operation")
    if operation == "optimize":
        for field in ("maxBootstrappedDemos", "maxLabeledDemos"):
            if type(config.get(field)) is not int or config[field] <= 0:
                raise ContractError(f"{field} must be a positive integer")
        if type(config.get("seed")) is not int:
            raise ContractError("seed must be an integer")
        threshold = config.get("metricThreshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ContractError("metricThreshold must be in (0, 1]")
        optimize(config)
    elif operation == "predict":
        predict(config)
    else:
        raise ContractError("Unknown DSPy operation")


if __name__ == "__main__":
    sys.excepthook = report_error
    main(sys.argv[1])
