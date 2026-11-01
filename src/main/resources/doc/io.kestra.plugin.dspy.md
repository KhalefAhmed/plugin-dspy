Run and optimize DSPy programs without imposing a business domain, retriever,
signature, input field or output field.

## Tasks

- `Predict` executes a declarative signature or a user-supplied Python program,
  directly or by reloading an artifact produced by `Optimize`.
- `Optimize` runs BootstrapFewShot with explicit student and teacher models,
  validates the candidate on a separate dataset, and returns artifact and manifest URIs.

## Program definitions

For declarative programs, set `signature` (for example `text -> category`),
optional `instructions`, and `predictor` (`PREDICT` or `CHAIN_OF_THOUGHT`).
`responseFormat` selects `CHAT` (DSPy field markers, the default for new programs)
or `JSON` (JSONAdapter, with native structured output when supported). Neither
mode retries parsing failures through another adapter. Artifacts retain the
format; reloading inherits it unless explicitly overridden. Older artifacts
without this field use CHAT.

For custom programs, set `programUri` to a trusted Python file. Its
`build_program()` function, or the function named by `programFactory`, must return
a `dspy.Module`. The same source file is required when reloading its artifact;
the plugin verifies its SHA-256 hash. Additional libraries belong in a custom
runtime image. Python code is executed, not sandboxed by the plugin.

## Data contract, version 1

Datasets are JSONL files with unique non-empty string IDs:

```json
{"schemaVersion":1,"id":"ticket-1","inputs":{"text":"Please send my invoice."},"reference":{"category":"billing"}}
```

`inputs` contains the keyword arguments passed to the program. `reference`
contains expected output fields. Prediction does not require references.
Training and validation IDs must be disjoint.

The default optimization metric is normalized exact match over every reference
field. Custom programs can provide a function named by `metricFunction`, with
signature `(example, prediction, trace=None)` returning a score in `[0, 1]`.
Only with a custom metric may references be omitted. Custom metrics access inputs
and references as fields on the DSPy example.

Predictions contain `schemaVersion`, `id`, `inputs`, `outputs` and `artifactVersion`.
No field named `question`, `contexts` or `answer` is imposed.

## Runner file paths

Source files are supplied through `datasetUri`, `trainsetUri`, `validationUri`,
`programUri` and `artifactUri`. Java copies and validates JSONL files without
retaining their records, then passes metadata and runner paths through a stored
configuration file. Dataset contents remain in the staged files and are never
rendered as templates. Python preflights prediction fields and reads each row
again for inference, without loading the whole prediction dataset. Optimize loads
the bounded training and validation files into memory for DSPy.

Use `filePaths` to override paths within the runner's working directory:

```yaml
filePaths:
  train: inputs/examples.jsonl
  validation: inputs/held-out.jsonl
  program: modules/my-program.py
  candidate: artifacts/compiled.json
  result: reports/evaluation.json
```

`Optimize` supports the roles shown above. `Predict` supports `program`, `dataset`,
`artifact`, `predictions` and `summary`. Omitted paths retain the existing defaults.
Paths must be distinct relative file paths, without glob patterns or traversal
outside the working directory. Output URI properties remain unchanged.

## Runtime and credentials

Build the default runtime image with:

```bash
docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .
```

Use Kestra secrets through `env`, such as `OPENAI_API_KEY`.
`DSPY_STUDENT_API_KEY`, `DSPY_TEACHER_API_KEY`, `DSPY_STUDENT_API_BASE` and
`DSPY_TEACHER_API_BASE` support distinct student and teacher connections.
Credentials are not written into the plugin's artifact or manifest.

Set `maxDatasetBytes` (default 67108864, or 64 MiB) and `maxDatasetRows` (default
10000) to cap each dataset separately. Oversized files fail before model calls.
ID tracking still uses memory proportional to the bounded row count.

Set `maxLlmCalls`, `maxTokens`, `requestTimeoutSeconds` and the Kestra task
`timeout` to bound execution. `maxRetries` defaults to 2 additional attempts for
rate limits, timeouts, transport errors and HTTP 500/502/503/504, with exponential
backoff capped at 8 seconds. Set it to 0 to disable retries. Provider-internal
retries remain disabled; every plugin retry consumes the same call budget.
The budget applies per task attempt: a Kestra retry starts a new task budget.
The call budget covers the plugin-supplied LMs,
including copies used by BootstrapFewShot; arbitrary requests in user Python
code are outside that budget. Errors fail the task rather than produce a zero
score or an incomplete successful batch.

Set `evaluate: true` on `Predict` to require consistent, non-empty references
and return an `evaluationScore` using normalized exact match. Compare baseline
and candidate scores on validation data before selecting a candidate. Keep the
test set out of that decision.

The default runner is Docker. Process requires Python and the locked dependencies
on the worker. Other TaskRunners require an accessible compatible image and
should be validated in the target environment.
