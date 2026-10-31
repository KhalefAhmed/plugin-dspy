package io.kestra.plugin.dspy;

import java.net.URI;
import java.util.List;
import java.util.Map;

import io.kestra.core.models.annotations.Example;
import io.kestra.core.models.annotations.Plugin;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTask;
import io.kestra.core.runners.RunContext;

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
@Schema(
    title = "Optimize a declarative or custom DSPy program",
    description = "Bootstraps demonstrations with an explicit teacher, evaluates the candidate on a disjoint validation set, and saves a JSON artifact plus manifest. Supports generic input/output fields and a metric supplied by the user."
)
@Plugin(examples = {
    @Example(title = "Build a candidate and predict", full = true, code = """
        id: dspy_optimize
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
            trainsetUri: "{{ inputs.train }}"
            validationUri: "{{ inputs.validation }}"
            studentModel: openai/gpt-4.1-mini
            teacherModel: openai/gpt-4.1
            signature: "text -> category"
            instructions: "Classify the text as billing, technical or other."
            env:
              OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
          - id: predict
            type: io.kestra.plugin.dspy.Predict
            artifactUri: "{{ outputs.optimize.candidateArtifactUri }}"
            datasetUri: "{{ inputs.test }}"
            env:
              OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
        """)
})
public class Optimize extends AbstractDspy implements RunnableTask<Optimize.Output> {

    public static final String METRIC_FUNCTION = "metricFunction";

    public enum Optimizer {
        BOOTSTRAP_FEW_SHOT
    }

    @NotNull
    @Schema(title = "URI of the training JSONL dataset", description = "Each record requires schemaVersion: 1, a unique id, inputs and reference objects. References may be omitted only with a custom metric.")
    private Property<String> trainsetUri;

    @NotNull
    @Schema(title = "URI of the disjoint validation JSONL dataset", description = "Uses the same format as training. Used only to assess the compiled candidate, not to fit demonstrations.")
    private Property<String> validationUri;

    @NotNull
    @Schema(title = "Student provider/model identifier")
    private Property<String> studentModel;

    @NotNull
    @Schema(title = "Teacher provider/model identifier", description = "The teacher is explicitly attached to the teacher program and passed to BootstrapFewShot.compile.")
    private Property<String> teacherModel;

    @NotNull
    @Builder.Default
    @Schema(title = "Optimization algorithm", description = "V1 supports BootstrapFewShot with one bootstrap round.")
    private Property<Optimizer> optimizer = Property.ofValue(Optimizer.BOOTSTRAP_FEW_SHOT);

    @NotNull
    @Builder.Default
    @Schema(title = "Random seed for training order", description = "Does not guarantee deterministic responses from a remote provider.")
    private Property<Integer> seed = Property.ofValue(42);

    @NotNull
    @Builder.Default
    @Schema(title = "Maximum bootstrapped demonstrations")
    private Property<Integer> maxBootstrappedDemos = Property.ofValue(4);

    @NotNull
    @Builder.Default
    @Schema(title = "Maximum labeled demonstrations")
    private Property<Integer> maxLabeledDemos = Property.ofValue(8);

    @Schema(title = "Custom metric function", description = "Name of a function in programUri accepting (example, prediction, trace=None) and returning a boolean or a finite score between 0 and 1. Defaults to normalized exact match on all reference fields.")
    private Property<String> metricFunction;

    @NotNull
    @Builder.Default
    @Schema(title = "Minimum bootstrap metric score", description = "Demonstrations are accepted at or above this threshold, in the interval (0, 1].")
    private Property<Double> metricThreshold = Property.ofValue(1.0);

    @Override
    public Output run(RunContext runContext) throws Exception {
        Map<String, Object> config = configuration(runContext);
        config.put("operation", "optimize");
        config.put("studentModel", required(runContext, studentModel, "studentModel"));
        config.put("teacherModel", required(runContext, teacherModel, "teacherModel"));
        config.put("optimizer", runContext.render(optimizer).as(Optimizer.class).orElseThrow().name());
        config.put("seed", runContext.render(seed).as(Integer.class).orElseThrow());
        config.put("maxBootstrappedDemos", positive(runContext, maxBootstrappedDemos, "maxBootstrappedDemos"));
        config.put("maxLabeledDemos", positive(runContext, maxLabeledDemos, "maxLabeledDemos"));
        if (runContext.render(metricFunction).as(String.class).isPresent()) {
            config.put(METRIC_FUNCTION, required(runContext, metricFunction, METRIC_FUNCTION));
        }
        double threshold = runContext.render(metricThreshold).as(Double.class).orElseThrow();
        if (!Double.isFinite(threshold) || threshold <= 0 || threshold > 1) {
            throw new IllegalArgumentException("metricThreshold must be in (0, 1]");
        }
        config.put("metricThreshold", threshold);
        Map<String, String> paths = filePaths(runContext, Map.of(
            "program", "user_program.py", "train", "train.jsonl", "validation", "validation.jsonl",
            "candidate", "candidate.json", "result", "optimization-result.json"
        ));
        Map<String, String> files = programFiles(runContext, paths);
        if (!config.containsKey("signature") && !files.containsKey("program")) {
            throw new IllegalArgumentException("Provide signature or programUri");
        }
        if (config.containsKey(METRIC_FUNCTION) && !files.containsKey("program")) {
            throw new IllegalArgumentException("metricFunction requires programUri");
        }
        files.put("train", uri(runContext, trainsetUri, "trainsetUri"));
        files.put("validation", uri(runContext, validationUri, "validationUri"));
        var result = execute(runContext, config, files, paths, List.of(paths.get("candidate"), paths.get("result")));
        URI candidateUri = outputUri(result, paths.get("candidate"));
        return Output.builder()
            .candidateArtifactUri(candidateUri)
            .manifestUri(DspyManifest.write(runContext, config, candidateUri, outputUri(result, paths.get("result"))))
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Candidate JSON artifact URI", description = "Contains the DSPy state without credentials or serialized executable code.")
        private final URI candidateArtifactUri;
        @Schema(title = "Manifest URI", description = "Contains versions, models, dataset hashes, seed, validation score and model-call counts.")
        private final URI manifestUri;
    }
}
