package io.kestra.plugin.dspy;

import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.kestra.core.exceptions.IllegalVariableEvaluationException;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.property.URIFetcher;
import io.kestra.core.runners.FilesService;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.plugin.scripts.exec.AbstractExecScript;
import io.kestra.plugin.scripts.exec.scripts.models.ScriptOutput;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@Getter
@ToString
@EqualsAndHashCode
@NoArgsConstructor
public abstract class AbstractDspy extends AbstractExecScript {
    public enum Predictor {
        PREDICT,
        CHAIN_OF_THOUGHT
    }

    public enum ResponseFormat {
        CHAT,
        JSON
    }

    @Schema(title = "Declarative DSPy signature", description = "For example `text -> category` or `question, contexts -> answer`. Mutually exclusive with programUri. Predict can instead load the definition from an artifact.")
    protected Property<String> signature;

    @Schema(title = "Instructions for the declarative signature", description = "Only supported with signature, not with a custom Python program.")
    protected Property<String> instructions;

    @Schema(title = "Declarative predictor", description = "Used with signature. A custom Python program may define any DSPy module.")
    @Builder.Default
    @NotNull
    protected Property<Predictor> predictor = Property.ofValue(Predictor.PREDICT);

    @Schema(title = "DSPy response format", description = "CHAT uses DSPy field markers; JSON uses JSONAdapter and provider structured output when supported. Defaults to CHAT for a new program, or the saved artifact's format when reloading. No automatic adapter retry.")
    protected Property<ResponseFormat> responseFormat;

    @Schema(title = "URI of a custom Python program", description = "The file must expose programFactory returning a dspy.Module. Custom artifacts require the same source file during prediction. Only run trusted code.")
    protected Property<String> programUri;

    @Schema(title = "Factory function in the custom Python file", description = "Called without arguments to construct the DSPy module. Use inputFiles, env or a custom image for additional resources and dependencies.")
    @Builder.Default
    @NotNull
    protected Property<String> programFactory = Property.ofValue("build_program");

    @Schema(
        title = "Runner file paths",
        description = "Overrides relative working-directory paths by role. Predict supports program, dataset, artifact, predictions and summary. Optimize supports program, train, validation, candidate and result. Paths may include subdirectories and must be distinct, without glob patterns or traversal outside the working directory. Source URIs remain configured with the task's URI properties."
    )
    @Builder.Default
    @NotNull
    protected Property<Map<String, String>> filePaths = Property.ofValue(Map.of());

    @Schema(
        title = "Image containing Python and DSPy",
        description = "Build `plugin-dspy:3.4.1` with `docker build -f python/Dockerfile -t plugin-dspy:3.4.1 .`, or override with your own compatible image."
    )
    @Builder.Default
    @NotNull
    protected Property<String> containerImage = Property.ofValue("plugin-dspy:3.4.1");

    @Schema(
        title = "Python executable",
        description = "For the Process runner, Python and the locked dependencies must already be installed on the worker. Linux runners are supported."
    )
    @Builder.Default
    @NotNull
    protected Property<String> pythonExecutable = Property.ofValue("python3");

    @Schema(title = "Maximum model calls across the task attempt", description = "Includes teacher, validation and retry attempts. Provider-internal retries, response caches and adapter fallbacks are disabled.")
    @Builder.Default
    @NotNull
    protected Property<Integer> maxLlmCalls = Property.ofValue(100);

    @Schema(title = "Maximum dataset size in bytes", description = "Applies separately to prediction, training and validation files. Enforced while downloading, before model calls. Default: 64 MiB.")
    @Builder.Default
    @NotNull
    protected Property<Long> maxDatasetBytes = Property.ofValue(DspyDataset.DEFAULT_MAX_BYTES);

    @Schema(title = "Maximum dataset rows", description = "Applies separately to prediction, training and validation files. Java validates rows without retaining their contents. Default: 10000.")
    @Builder.Default
    @NotNull
    protected Property<Integer> maxDatasetRows = Property.ofValue(DspyDataset.DEFAULT_MAX_ROWS);

    @Schema(title = "Maximum additional model-request retries", description = "Default: 2. Only transient failures are retried with bounded exponential backoff. Every attempt consumes maxLlmCalls. Set 0 to disable retries.")
    @Builder.Default
    @NotNull
    protected Property<Integer> maxRetries = Property.ofValue(2);

    @Schema(title = "Timeout per model request, in seconds")
    @Builder.Default
    @NotNull
    protected Property<Integer> requestTimeoutSeconds = Property.ofValue(60);

    @Schema(title = "Maximum output tokens per model request")
    @Builder.Default
    @NotNull
    protected Property<Integer> maxTokens = Property.ofValue(1024);

    protected Map<String, Object> configuration(RunContext runContext) throws IllegalVariableEvaluationException {
        Map<String, Object> configuration = new LinkedHashMap<>();
        configuration.put("schemaVersion", 1);
        configuration.put("maxLlmCalls", positive(runContext, maxLlmCalls, "maxLlmCalls"));
        configuration.put("maxDatasetRows", positive(runContext, maxDatasetRows, "maxDatasetRows"));
        long byteLimit = runContext.render(maxDatasetBytes).as(Long.class).orElseThrow(
            () -> new IllegalArgumentException("maxDatasetBytes is required")
        );
        if (byteLimit <= 0) {
            throw new IllegalArgumentException("maxDatasetBytes must be positive");
        }
        configuration.put("maxDatasetBytes", byteLimit);
        int retries = runContext.render(maxRetries).as(Integer.class).orElseThrow(
            () -> new IllegalArgumentException("maxRetries is required")
        );
        if (retries < 0) {
            throw new IllegalArgumentException("maxRetries must not be negative");
        }
        configuration.put("maxRetries", retries);
        configuration.put("requestTimeoutSeconds", positive(runContext, requestTimeoutSeconds, "requestTimeoutSeconds"));
        configuration.put("maxTokens", positive(runContext, maxTokens, "maxTokens"));
        runContext.render(signature).as(String.class).ifPresent(value -> configuration.put("signature", value));
        runContext.render(instructions).as(String.class).ifPresent(value -> configuration.put("instructions", value));
        configuration.put("predictor", runContext.render(predictor).as(Predictor.class).orElseThrow().name());
        runContext.render(responseFormat).as(ResponseFormat.class).ifPresent(
            format -> configuration.put("responseFormat", format.name())
        );
        configuration.put("programFactory", required(runContext, programFactory, "programFactory"));
        return configuration;
    }

    protected Map<String, String> programFiles(RunContext context, Map<String, String> paths) throws IllegalVariableEvaluationException {
        Map<String, String> files = new LinkedHashMap<>();
        if (context.render(programUri).as(String.class).isPresent()) {
            if (context.render(signature).as(String.class).isPresent() || context.render(instructions).as(String.class).isPresent()) {
                throw new IllegalArgumentException("programUri cannot be combined with signature or instructions");
            }
            files.put("program", uri(context, programUri, "programUri"));
        } else {
            paths.remove("program");
        }
        return files;
    }

    protected Map<String, String> getfilePaths(RunContext context, Map<String, String> defaults) throws IllegalVariableEvaluationException {
        Map<String, String> paths = new LinkedHashMap<>(defaults);
        Map<String, String> overrides = context.render(filePaths).asMap(String.class, String.class);
        for (var entry : overrides.entrySet()) {
            if (!defaults.containsKey(entry.getKey())) {
                throw new IllegalArgumentException("Unsupported filePaths role: " + entry.getKey());
            }
            paths.put(entry.getKey(), entry.getValue());
        }
        paths.replaceAll(AbstractDspy::runnerPath);
        if (new HashSet<>(paths.values()).size() != paths.size()) {
            throw new IllegalArgumentException("filePaths must use distinct paths");
        }
        return paths;
    }

    private static String runnerPath(String role, String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("filePaths." + role + " must not be blank");
        }
        Path path = Path.of(value).normalize();
        if (path.isAbsolute() || path.startsWith("..") || path.toString().isEmpty()
            || value.chars().anyMatch(character -> "\\*?[]{}".indexOf(character) >= 0)) {
            throw new IllegalArgumentException("filePaths." + role + " must be a relative file path without glob patterns");
        }
        String name = path.toString();
        if (name.equals("dspy_task.py") || name.equals("dspy_config.json")) {
            throw new IllegalArgumentException("Reserved runner filename: " + name);
        }
        return name;
    }

    protected int positive(RunContext context, Property<Integer> property, String name) throws IllegalVariableEvaluationException {
        int value = context.render(property).as(Integer.class).orElseThrow(() -> new IllegalArgumentException(name + " is required"));
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }

    protected String required(RunContext context, Property<String> property, String name) throws IllegalVariableEvaluationException {
        String value = context.render(property).as(String.class).orElseThrow(() -> new IllegalArgumentException(name + " is required"));
        if (value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }

    protected String uri(RunContext context, Property<String> property, String name) throws IllegalVariableEvaluationException {
        String value = required(context, property, name);
        URIFetcher.of(value);
        return value;
    }

    protected ScriptOutput execute(RunContext context, Map<String, Object> configuration, Map<String, String> datasets, Map<String, String> paths, List<String> outputs) throws Exception {
        var commands = commands(context);
        Map<String, String> inputs = new LinkedHashMap<>(
            FilesService.inputFiles(context, commands.getTaskRunner().additionalVars(context, commands), getInputFiles())
        );
        for (String reserved : paths.values()) {
            if (inputs.containsKey(reserved)) {
                throw new IllegalArgumentException("Reserved input filename: " + reserved);
            }
        }
        configuration.put("files", paths);
        Map<String, String> internal = DspyInputs.prepare(context, configuration, datasets, paths);
        internal.put("dspy_task.py", resource());
        var configFile = context.workingDir().createTempFile(JacksonMapper.ofJson().writeValueAsBytes(configuration));
        internal.put("dspy_config.json", context.storage().putFile(configFile.toFile()).toString());
        for (var entry : internal.entrySet()) {
            if (inputs.putIfAbsent(entry.getKey(), entry.getValue()) != null) {
                throw new IllegalArgumentException("Reserved input filename: " + entry.getKey());
            }
        }

        String executable = required(context, pythonExecutable, "pythonExecutable");
        List<String> captured = new ArrayList<>(context.render(getOutputFiles()).asList(String.class));
        captured.addAll(outputs);
        Map<String, String> environment = new LinkedHashMap<>(commands.getEnv());
        environment.put("PYTHONUNBUFFERED", "1");
        environment.put("DSPY_LOG_LEVEL", "ERROR");
        environment.putIfAbsent("LITELLM_LOCAL_MODEL_COST_MAP", "True");
        ScriptOutput result = commands
            .withInputFiles(inputs)
            .withOutputFiles(captured)
            .withInterpreter(interpreter)
            .withBeforeCommands(beforeCommands)
            .withBeforeCommandsWithOptions(true)
            .withEnv(environment)
            .withCommands(Property.ofValue(List.of("'" + executable.replace("'", "'\\''") + "' dspy_task.py dspy_config.json")))
            .run();
        if (result.getExitCode() != 0) {
            throw new IllegalStateException("DSPy runner exited with code " + result.getExitCode());
        }
        for (String output : outputs) {
            outputUri(result, output);
        }
        return result;
    }

    protected URI outputUri(ScriptOutput output, String name) {
        URI uri = output.getOutputFiles() == null ? null : output.getOutputFiles().get(name);
        if (uri == null) {
            throw new IllegalStateException("DSPy did not produce " + name);
        }
        return uri;
    }

    private String resource() throws IOException {
        try (var stream = getClass().getResourceAsStream("/scripts/dspy_task.py")) {
            if (stream == null) {
                throw new IOException("Missing embedded resource: " + "/scripts/dspy_task.py");
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
