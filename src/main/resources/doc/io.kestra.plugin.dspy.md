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

## Runtime and credentials

Build the default runtime image with:

```bash
docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .
```

Use Kestra secrets through `env`, such as `OPENAI_API_KEY`.
`DSPY_STUDENT_API_KEY`, `DSPY_TEACHER_API_KEY`, `DSPY_STUDENT_API_BASE` and
`DSPY_TEACHER_API_BASE` support distinct student and teacher connections.
Credentials are not written into the plugin's artifact or manifest.

Set `maxLlmCalls`, `maxTokens`, `requestTimeoutSeconds` and the Kestra task
`timeout` to bound execution. The call budget covers the plugin-supplied LMs,
including copies used by BootstrapFewShot; arbitrary requests in user Python
code are outside that budget. Errors fail the task rather than produce a zero
score or an incomplete successful batch.

The default runner is Docker. Process requires Python and the locked dependencies
on the worker. Other TaskRunners require an accessible compatible image and
should be validated in the target environment.
