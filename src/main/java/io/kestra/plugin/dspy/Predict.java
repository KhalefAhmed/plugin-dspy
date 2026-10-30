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
    title = "Run a declarative or custom DSPy program",
    description = "Runs a signature or a user-supplied Python module on generic JSONL inputs. Optionally reloads an artifact produced by Optimize, without recompilation."
)
@Plugin(examples = {
    @Example(title = "Classify text with a declarative signature", full = true, code = """
        id: dspy_predict
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
            model: openai/gpt-4.1-mini
            env:
              OPENAI_API_KEY: "{{ secret('OPENAI_API_KEY') }}"
        """)
})
public class Predict extends AbstractDspy implements RunnableTask<Predict.Output> {
    @Schema(title = "Optional URI of a JSON artifact produced by Optimize", description = "Without an artifact, provide signature or programUri and model. A custom artifact also requires its original programUri.")
    private Property<String> artifactUri;

    @NotNull
    @Schema(title = "URI of the prediction JSONL dataset", description = "Each record requires schemaVersion: 1, a unique id and an inputs object whose keys match the program's arguments. References are not required.")
    private Property<String> datasetUri;

    @Schema(title = "Provider/model identifier", description = "Required for direct prediction. With an artifact, defaults to its student model and can be overridden.")
    private Property<String> model;

    @Override
    public Output run(RunContext runContext) throws Exception {
        Map<String, Object> config = configuration(runContext);
        config.put("operation", "predict");
        runContext.render(model).as(String.class).ifPresent(value -> config.put("model", value));
        Map<String, String> files = programFiles(runContext);
        files.put("dataset.jsonl", uri(runContext, datasetUri, "datasetUri"));
        if (runContext.render(artifactUri).as(String.class).isPresent()) {
            if (config.containsKey("signature") || config.containsKey("instructions")) {
                throw new IllegalArgumentException("An artifact already defines its signature and instructions");
            }
            files.put("artifact.json", uri(runContext, artifactUri, "artifactUri"));
        } else if (!config.containsKey("signature") && !files.containsKey("user_program.py")) {
            throw new IllegalArgumentException("Provide artifactUri, signature or programUri");
        }
        var result = execute(runContext, config, files, List.of("predictions.jsonl", "prediction-summary.json"));
        return Output.builder()
            .predictionsUri(outputUri(result, "predictions.jsonl"))
            .summaryUri(outputUri(result, "prediction-summary.json"))
            .build();
    }

    @Builder
    @Getter
    public static class Output implements io.kestra.core.models.tasks.Output {
        @Schema(title = "Predictions JSONL URI", description = "Records include id, inputs, outputs and artifactVersion.")
        private final URI predictionsUri;
        @Schema(title = "Prediction summary URI", description = "Contains the artifact version, model, number of records and model calls.")
        private final URI summaryUri;
    }
}
