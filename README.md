# DSPy plugin for Kestra

Run and optimize **user-defined DSPy programs** in Kestra. The plugin does not
impose a RAG pipeline, retriever, signature or business-specific input/output.

| Task | Purpose |
|---|---|
| `io.kestra.plugin.dspy.Predict` | Execute a signature or Python module, directly or from a saved artifact. |
| `io.kestra.plugin.dspy.Optimize` | Bootstrap demonstrations with an explicit teacher, validate and save a candidate. |

## Build and installation

Requirements: **JDK 25**, Docker and the included Gradle wrapper.

```bash
docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .
DSPY_DOCKER_TESTS=true ./gradlew test lintPluginDocs shadowJar
docker compose up --build
```

The JAR is generated in `build/libs/`. The supplied Compose configuration loads it
into Kestra **2.0.7**, available at <http://localhost:8080>. It mounts the Docker
socket to allow Docker TaskRunners: use this configuration only for trusted local
development. Mounting `build/libs/` over `/app/plugins/` hides additional plugin
JARs bundled at that location in the base image.

For an existing Kestra deployment, install the JAR on the workers and other
components that load plugin definitions. Build the runtime image on the worker's
Docker daemon, or push it to a registry and configure `containerImage`.

The runtime uses **DSPy 3.4.1** and **Python 3.14.10**. Python 3.15 was attempted but
native dependencies did not build on ARM64 without additional compiler support.
The base image is pinned by digest and Python dependencies by
`python/requirements.lock`. To update dependencies, resolve `python/requirements.txt`
in a clean compatible environment, regenerate the lock and rerun the checks.

In IntelliJ, select JDK 25 for both the project SDK and Gradle JVM, and enable
annotation processing.

## Generic dataset format

Use JSONL records with `schemaVersion: 1`, a unique string `id`, an `inputs` object,
and optionally a `reference` object:

```json
{"schemaVersion":1,"id":"ticket-1","inputs":{"text":"Please send my invoice."},"reference":{"category":"billing"}}
```

`inputs` maps directly to the program's keyword arguments. `reference` contains
the expected output fields. Classification, extraction, translation and RAG use
the same envelope, with different field names inside those objects.

For declarative programs, fields must match the signature. Typed signatures use
DSPy's string syntax, such as `text -> entities: list[str]`. Custom programs define
their own argument and output types.

Task URI properties accept Kestra storage or namespace file URIs. Download remote
files with a separate Kestra task first; do not put JSONL contents directly into
URI properties.

## Mode 1: declarative signature

Upload a prediction dataset without references, then run:

```yaml
id: classify
namespace: company.team
inputs:
  - id: dataset
    type: FILE
tasks:
  - id: predict
    type: io.kestra.plugin.dspy.Predict
    datasetUri: "{{ inputs.dataset }}"
    signature: "text -> category"
    instructions: "Classify the text as billing, technical or other."
    predictor: PREDICT
    model: openai/gpt-4.1-mini
    maxLlmCalls: 100
    timeout: PT10M
    env:
      OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
```

Use `CHAIN_OF_THOUGHT` instead of `PREDICT` for a reasoning predictor.
Predictions are written to `outputs.predict.predictionsUri`:

```json
{"schemaVersion":1,"id":"ticket-1","inputs":{"text":"Please send my invoice."},"outputs":{"category":"billing"},"artifactVersion":"..."}
```

The summary URI reports the model, row count, execution duration and model calls.

## Mode 2: custom Python program

Supply a trusted Python file, for example:

```python
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
    return prediction.translation.strip() == example.translation.strip()
```

Then use `programUri` instead of `signature`:

```yaml
id: translate
namespace: company.team
inputs:
  - id: program
    type: FILE
  - id: dataset
    type: FILE
tasks:
  - id: predict
    type: io.kestra.plugin.dspy.Predict
    programUri: "{{ inputs.program }}"
    programFactory: build_program
    datasetUri: "{{ inputs.dataset }}"
    model: openai/gpt-4.1-mini
    env:
      OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
```

The dataset can use arbitrary domain fields, for example:

```json
{"schemaVersion":1,"id":"translation-1","inputs":{"sentence":"Bonjour"},"reference":{"translation":"Hello"}}
```

The factory takes no arguments and returns a synchronous `dspy.Module`. Configure
its input fields through the dataset, additional files through inherited
`inputFiles`, and settings through `env`. Install additional libraries in a custom
image set with `containerImage`. The plugin attaches the configured LM to the
module's predictors. Output values must be JSON serializable.

**Custom code is executed with the runner's permissions.** Only run trusted
programs. The plugin does not provide a Python sandbox, and user code is
responsible for its own external calls, logging and secret handling.

## Optimize and reuse

`Optimize` accepts either program definition mode. Its V1 optimizer is
`BOOTSTRAP_FEW_SHOT`. The validation set assesses the candidate after compilation;
it is not used to fit demonstrations.

```yaml
id: optimize_and_predict
namespace: company.team
inputs:
  - id: train
    type: FILE
  - id: validation
    type: FILE
  - id: test
    type: FILE
tasks:
  - id: optimize
    type: io.kestra.plugin.dspy.Optimize
    signature: "text -> category"
    instructions: "Classify the text as billing, technical or other."
    trainsetUri: "{{ inputs.train }}"
    validationUri: "{{ inputs.validation }}"
    studentModel: openai/gpt-4.1-mini
    teacherModel: openai/gpt-4.1
    optimizer: BOOTSTRAP_FEW_SHOT
    seed: 42
    maxLlmCalls: 100
    timeout: PT15M
    env:
      OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
  - id: predict
    type: io.kestra.plugin.dspy.Predict
    artifactUri: "{{ outputs.optimize.candidateArtifactUri }}"
    datasetUri: "{{ inputs.test }}"
    env:
      OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
```

Keep train, validation and test IDs disjoint. The plugin rejects overlapping
train/validation IDs and duplicate IDs within a dataset.

The default metric compares every reference field: strings use Unicode NFKC,
case folding and collapsed whitespace; other JSON values use exact equality.
Default-metric datasets require consistent, non-empty `reference` objects.

For custom programs, set `metricFunction` to a function in `programUri` with
signature `(example, prediction, trace=None)` returning a boolean or finite
number in `[0, 1]`. Example fields combine `inputs` and `reference`; the two objects
must not use overlapping keys. Only with a custom metric can references be omitted.
`metricThreshold` controls accepted bootstrap demonstrations, defaulting to `1.0`.

`Optimize` outputs:

- `candidateArtifactUri`: versioned JSON state, not pickle or executable code.
- `manifestUri`: program definition/source hash, runtime versions, models, dataset
  hashes, optimizer settings, validation score and model-call counts.

An artifact records its program definition and student model. Prediction verifies
its integrity and DSPy version and reloads the same module structure without
recompilation. Custom artifacts additionally require `programUri` containing
the original source file; a different hash is rejected. Artifacts can contain
training examples, so handle them with the same confidentiality as the datasets.
For long-term use, retain the matching runtime image and custom dependencies.

## Credentials, budgets and runners

Use Kestra secrets through `env`. Provider-native variables such as
`OPENAI_API_KEY` work. Distinct connections can be configured through:

- `DSPY_STUDENT_API_KEY` and `DSPY_STUDENT_API_BASE`.
- `DSPY_TEACHER_API_KEY` and `DSPY_TEACHER_API_BASE`.

The plugin does not serialize its LM credentials into artifacts or manifests.
Provider errors are surfaced with their error class but without full payloads.

`maxLlmCalls` (default 100), `maxTokens` (1024) and `requestTimeoutSeconds` (60)
bound the plugin-supplied LMs. Retries, response caching and adapter fallback
calls are disabled. Set the inherited task `timeout` as an overall deadline.
Calls made independently by custom Python code are not governed by this budget.
Model-specific requirements still apply, including minimum token budgets for
some reasoning models. Seeds do not make remote providers deterministic.

The default TaskRunner is Docker. With Process, install compatible Python and
`python/requirements.lock` on the worker, and optionally set `pythonExecutable`.
Other TaskRunners require a registry-accessible compatible image and should be
validated in the target infrastructure. The embedded launcher currently targets
Linux/POSIX execution.

Files named `dspy_task.py`, `dspy_config.json`, `user_program.py`, `artifact.json`,
`examples/dataset.jsonl`, `train.jsonl` and `validation.jsonl` are reserved task inputs.

## Checks

Without Docker, run `./gradlew test lintPluginDocs shadowJar`; Docker integration
tests are opt-in. To cover the runtime and Docker integration:

```bash
docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .
docker run --rm --network none -v "$PWD:/workspace:ro" -w /workspace \
  plugin-dspy:3.4.1 python -m unittest discover -s python -v
DSPY_DOCKER_TESTS=true ./gradlew test lintPluginDocs shadowJar
```

These checks use deterministic fake LLM endpoints, not paid providers. The CI
setup hook builds the image and enables both Python and Docker checks.

## License

Apache License 2.0; see [LICENSE](LICENSE).
