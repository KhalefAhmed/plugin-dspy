# Kestra DSPy Plugin

## What

- Provides plugin components under `io.kestra.plugin.dspy`.
- Provides `Predict` and `Optimize` tasks.

## Why

- Execute and optimize DSPy programs without a hard-coded business use case.
- Users can provide a declarative signature or a trusted Python module factory.
- Datasets use generic `inputs` and optional `reference` objects, not RAG-specific fields.

## How

### Architecture

Single-module plugin. Source packages under `io.kestra.plugin`:

- `dspy`

Infrastructure dependencies (Docker Compose services):

- `app`

### Key Plugin Classes

- `io.kestra.plugin.dspy.Predict`
- `io.kestra.plugin.dspy.Optimize`
- `AbstractDspy` reuses Kestra's `AbstractExecScript` and TaskRunner APIs.
- Python execution is implemented in `src/main/resources/scripts/dspy_task.py`.

### Project Structure

```
plugin-dspy/
├── src/main/java/io/kestra/plugin/dspy/
├── src/test/java/io/kestra/plugin/dspy/
├── build.gradle
└── README.md
```

## Local rules

- Base the wording on the implemented packages and classes, not on template README text.
- Keep tasks generic: domain-specific fields belong in user signatures or programs.
- Runtime image dependencies are locked in `python/requirements.lock`.
- Build the runtime image, then use `DSPY_DOCKER_TESTS=true ./gradlew test lintPluginDocs shadowJar` for integration checks.

## References

- https://kestra.io/docs/plugin-developer-guide
- https://kestra.io/docs/plugin-developer-guide/contribution-guidelines
