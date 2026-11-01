package io.kestra.plugin.dspy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.services.FlowParsingService;
import io.kestra.core.utils.TestsUtils;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@KestraTest
class DspyTest {
    @Inject
    RunContextFactory runContextFactory;

    @Inject
    FlowParsingService flowParsingService;

    @TempDir
    Path directory;

    @Test
    void validatesDatasetContractsInJava() throws Exception {
        String valid = """
            {"schemaVersion":1,"id":"one","inputs":{"text":"invoice"},"reference":{"category":"billing"}}
            """;
        Path dataset = directory.resolve("dataset.jsonl");
        for (String invalid : new String[] {
            "", valid + valid, "not-json\n", valid.replace("\"inputs\":{\"text\":\"invoice\"}", "\"inputs\":[]"),
            valid.replace("\"schemaVersion\":1", "\"schemaVersion\":true"),
            valid.replace("\"schemaVersion\":1", "\"schemaVersion\":1.0"),
            valid.replace("\"one\"", "\"\\u00a0\""),
            valid.replace("\"category\"", "\"text\""),
            valid.replace("{\"category\":\"billing\"}", "{}"),
            valid.replace("{\"category\":\"billing\"}", "null"),
            valid.replace("\"invoice\"", "1e999"),
            valid.strip() + " {}\n"
        }) {
            Files.writeString(dataset, invalid);
            assertThrows(IllegalArgumentException.class, () -> DspyDataset.read(dataset, true));
        }
        Files.writeString(dataset, valid);
        DspyDataset validated = DspyDataset.read(dataset, true);
        assertEquals(Set.of("one"), validated.ids());
        assertEquals(Set.of(Set.of("category")), validated.referenceFields());
        assertEquals(DspyDataset.digest(dataset), validated.sha256());
        Files.writeString(dataset, """
            {"schemaVersion":1,"id":"test","inputs":{"text":"{{ keep this literal }}"}}
            """);
        assertEquals(Set.of(Set.of()), DspyDataset.read(dataset, false).referenceFields());
        assertThrows(IllegalArgumentException.class, () -> DspyDataset.read(dataset, true));
    }

    @Test
    void rejectsOverlappingAndInconsistentOptimizationDatasetsBeforeTheRunner() throws Exception {
        Optimize task = Optimize.builder()
            .id("optimize")
            .type(Optimize.class.getName())
            .signature(Property.ofValue("text -> category"))
            .studentModel(Property.ofValue("openai/student"))
            .teacherModel(Property.ofValue("openai/teacher"))
            .trainsetUri(Property.ofExpression("{{ inputs.train }}"))
            .validationUri(Property.ofExpression("{{ inputs.validation }}"))
            .build();
        var uploadContext = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        Path train = directory.resolve("train.jsonl");
        String row = """
            {"schemaVersion":1,"id":"train","inputs":{"text":"invoice"},"reference":{"category":"billing"}}
            """;
        Files.writeString(train, row);
        String trainUri = uploadContext.storage().putFile(train.toFile()).toString();
        for (String validationRow : new String[] {row, row.replace("\"train\"", "\"validation\"").replace("\"category\"", "\"answer\"")}) {
            Path validation = directory.resolve("validation.jsonl");
            Files.writeString(validation, validationRow);
            String validationUri = uploadContext.storage().putFile(validation.toFile()).toString();
            var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of("train", trainUri, "validation", validationUri));
            var error = assertThrows(IllegalArgumentException.class, () -> task.run(context));
            assertTrue(error.getMessage().contains("disjoint") || error.getMessage().contains("same output fields"));
        }
    }

    @Test
    void derivesOptimizationReferenceFieldsFromArbitraryUserData() throws Exception {
        Optimize task = Optimize.builder().id("optimize").type(Optimize.class.getName()).build();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        String row = """
            {"schemaVersion":1,"id":"train","inputs":{"source":"bonjour","locale":"en"},
             "reference":{"translation":"hello","confidence":0.9}}
            """.replace("\n", "") + "\n";
        Path train = directory.resolve("train.jsonl");
        Path validation = directory.resolve("validation.jsonl");
        Files.writeString(train, row);
        Files.writeString(validation, row.replace("\"train\"", "\"validation\""));
        Map<String, Object> config = new LinkedHashMap<>(Map.of(
            "operation", "optimize", "signature", "source, locale -> translation, confidence: float",
            "predictor", "PREDICT"
        ));
        DspyInputs.prepare(context, config, Map.of(
            "train", context.storage().putFile(train.toFile()).toString(),
            "validation", context.storage().putFile(validation.toFile()).toString()
        ), Map.of("train", "inputs/custom-training.jsonl", "validation", "inputs/custom-validation.jsonl"));
        assertEquals(Set.of("translation", "confidence"), config.get("referenceFields"));
        assertFalse(config.containsKey("datasets"));
        var metadata = io.kestra.core.serializers.JacksonMapper.ofJson().valueToTree(config.get("train"));
        assertEquals(1, metadata.get("count").asInt());
        var program = io.kestra.core.serializers.JacksonMapper.ofJson().valueToTree(config.get("program"));
        assertEquals(config.get("signature"), program.get("signature").asText());
    }

    @Test
    void stagesExactSourceBytesAndBuildsCustomProgramDescriptor() throws Exception {
        Predict task = Predict.builder().id("predict").type(Predict.class.getName()).build();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        Path dataset = directory.resolve("dataset.jsonl");
        String contents = "{\"schemaVersion\":1,\"id\":\"test\",\"inputs\":{\"text\":\"{{ not a template }}\"}}\r\n";
        Files.writeString(dataset, contents);
        Path program = directory.resolve("program.py");
        Files.writeString(program, "def build_program():\n    pass\n");
        String sourceHash = DspyDataset.digest(program);
        Map<String, Object> config = new LinkedHashMap<>(Map.of("operation", "predict", "programFactory", "build_program"));
        var staged = DspyInputs.prepare(context, config, Map.of(
            "dataset", context.storage().putFile(dataset.toFile()).toString(),
            "program", context.storage().putFile(program.toFile()).toString()
        ), Map.of("dataset", "inputs/custom-dataset.jsonl", "program", "modules/custom-program.py"));
        assertEquals(Map.of("kind", "python", "factory", "build_program", "sha256", sourceHash), config.get("program"));
        assertFalse(config.containsKey("datasets"));
        var metadata = io.kestra.core.serializers.JacksonMapper.ofJson().valueToTree(config.get("dataset"));
        assertEquals(1, metadata.get("count").asInt());
        try (var stream = context.storage().getFile(java.net.URI.create(staged.get("inputs/custom-dataset.jsonl")))) {
            assertEquals(contents, new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
        }
        assertFalse(config.containsKey("referenceFields"));
    }

    @Test
    void rejectsInvalidRunnerPathsBeforeFetchingFiles() {
        for (Map<String, String> paths : java.util.List.of(
            Map.of("unknown", "input.jsonl"),
            Map.of("dataset", ""),
            Map.of("dataset", "/outside.jsonl"),
            Map.of("dataset", "../outside.jsonl"),
            Map.of("dataset", "inputs/*.jsonl"),
            Map.of("dataset", "inputs/\n*.jsonl"),
            Map.of("dataset", "dspy_task.py"),
            Map.of("dataset", "dspy_config.json"),
            Map.of("dataset", "shared.jsonl", "predictions", "shared.jsonl")
        )) {
            Predict task = Predict.builder()
                .id("predict").type(Predict.class.getName())
                .signature(Property.ofValue("source -> answer"))
                .model(Property.ofValue("openai/student"))
                .datasetUri(Property.ofValue("kestra:///dataset.jsonl"))
                .filePaths(Property.ofValue(paths))
                .build();
            var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
            assertThrows(IllegalArgumentException.class, () -> task.run(context));
        }
    }

    @Test
    void rejectsInlineDataInsteadOfAnArtifactUri() {
        Predict task = Predict.builder()
            .id("predict")
            .type(Predict.class.getName())
            .artifactUri(Property.ofValue("{\"state\": {}}"))
            .datasetUri(Property.ofValue("kestra:///dataset.jsonl"))
            .build();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        assertThrows(IllegalArgumentException.class, () -> task.run(context));
    }

    @Test
    void rejectsInvalidBudgetBeforeStartingTheRunner() {
        Optimize task = Optimize.builder()
            .id("optimize")
            .type(Optimize.class.getName())
            .maxLlmCalls(Property.ofValue(0))
            .studentModel(Property.ofValue("openai/student"))
            .teacherModel(Property.ofValue("openai/teacher"))
            .trainsetUri(Property.ofValue("kestra:///train.jsonl"))
            .validationUri(Property.ofValue("kestra:///validation.jsonl"))
            .build();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        var error = assertThrows(IllegalArgumentException.class, () -> task.run(context));
        assertTrue(error.getMessage().contains("maxLlmCalls"));
    }

    @Test
    void enforcesDatasetLimitsAtExactBoundariesAndDuringStaging() throws Exception {
        String row = "{\"schemaVersion\":1,\"id\":\"one\",\"inputs\":{\"text\":\"\u00e9\"}}\n";
        byte[] bytes = row.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        Path dataset = directory.resolve("bounded.jsonl");
        Files.write(dataset, bytes);
        assertEquals(Set.of("one"), DspyDataset.read(dataset, "bounded.jsonl", false, bytes.length, 1).ids());
        assertThrows(IllegalArgumentException.class, () -> DspyDataset.read(dataset, "bounded.jsonl", false, bytes.length - 1, 1));
        Files.writeString(dataset, row + row.replace("\"one\"", "\"two\""));
        long size = Files.size(dataset);
        assertEquals(2, DspyDataset.read(dataset, "bounded.jsonl", false, size, 2).ids().size());
        assertThrows(IllegalArgumentException.class, () -> DspyDataset.read(dataset, "bounded.jsonl", false, size, 1));

        Predict task = Predict.builder().id("predict").type(Predict.class.getName()).build();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        String uri = context.storage().putFile(dataset.toFile()).toString();
        Map<String, String> files = Map.of("dataset", uri);
        Map<String, String> locations = Map.of("dataset", "bounded.jsonl");
        for (Map<String, Object> limits : java.util.List.<Map<String, Object>>of(
            Map.of("maxDatasetBytes", size - 1, "maxDatasetRows", 2),
            Map.of("maxDatasetBytes", size, "maxDatasetRows", 1)
        )) {
            Map<String, Object> config = new LinkedHashMap<>(limits);
            config.putAll(Map.of("operation", "predict", "signature", "text -> category", "predictor", "PREDICT"));
            assertThrows(IllegalArgumentException.class, () -> DspyInputs.prepare(context, config, files, locations));
        }
    }

    @Test
    void rejectsInvalidLimitsAndRetriesBeforeFetchingInputs() {
        for (Predict task : java.util.List.of(
            Predict.builder().id("bytes").type(Predict.class.getName())
                .datasetUri(Property.ofValue("kestra:///unused.jsonl")).maxDatasetBytes(Property.ofValue(0L)).build(),
            Predict.builder().id("rows").type(Predict.class.getName())
                .datasetUri(Property.ofValue("kestra:///unused.jsonl")).maxDatasetRows(Property.ofValue(0)).build(),
            Predict.builder().id("retries").type(Predict.class.getName())
                .datasetUri(Property.ofValue("kestra:///unused.jsonl")).maxRetries(Property.ofValue(-1)).build()
        )) {
            var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
            assertThrows(IllegalArgumentException.class, () -> task.configuration(context));
        }
    }

    @Test
    void evaluationRequiresReferencesBeforeStartingTheRunner() throws Exception {
        Predict task = Predict.builder()
            .id("evaluate").type(Predict.class.getName())
            .signature(Property.ofValue("text -> category")).model(Property.ofValue("openai/student"))
            .evaluate(Property.ofValue(true))
            .datasetUri(Property.ofExpression("{{ inputs.dataset }}"))
            .build();
        var uploadContext = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        Path dataset = directory.resolve("unlabelled.jsonl");
        Files.writeString(dataset, "{\"schemaVersion\":1,\"id\":\"one\",\"inputs\":{\"text\":\"invoice\"}}\n");
        String uri = uploadContext.storage().putFile(dataset.toFile()).toString();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of("dataset", uri));
        var error = assertThrows(IllegalArgumentException.class, () -> task.run(context));
        assertTrue(error.getMessage().contains("reference"));
    }

    @Test
    void candidatePromotionUsesValidationAndFallsBackWithoutCandidateOutputs() throws Exception {
        String source = Files.readString(Path.of("examples/ollama-classification/flow.yaml"));
        flowParsingService.parse(null, source, true);
        var flow = io.kestra.core.serializers.JacksonMapper.ofYaml().readTree(source);
        String condition = flow.get("tasks").get(4).get("condition").asText();
        String selected = flow.get("outputs").get(1).get("value").asText();
        Predict task = Predict.builder().id("selection").type(Predict.class.getName())
            .datasetUri(Property.ofValue("kestra:///unused.jsonl")).build();
        var context = TestsUtils.mockRunContext(runContextFactory, task, Map.of());
        for (double[] scores : new double[][] {
            {0.7, 0.5, 0}, {0.9, 1.0, 0}, {0.9, 0.8, 1}, {0.8, 0.8, 1}
        }) {
            boolean accepted = scores[2] == 1;
            Map<String, Object> outputs = new LinkedHashMap<>(Map.of(
                "candidate_validation", Map.of("evaluationScore", scores[0]),
                "baseline_validation", Map.of("evaluationScore", scores[1]),
                "baseline", Map.of("predictionsUri", "kestra:///baseline.jsonl"),
                "select_candidate", Map.of("evaluationResult", accepted)
            ));
            if (accepted) {
                outputs.put("optimized", Map.of("predictionsUri", "kestra:///candidate.jsonl"));
            }
            Map<String, Object> variables = Map.of(
                "outputs", outputs, "inputs", Map.of("minimum_validation_score", 0.8f)
            );
            assertEquals(Boolean.toString(accepted), context.render(condition, variables));
            assertEquals(accepted ? "kestra:///candidate.jsonl" : "kestra:///baseline.jsonl", context.render(selected, variables));
        }
    }
}
