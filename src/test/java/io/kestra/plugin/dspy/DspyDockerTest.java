package io.kestra.plugin.dspy;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

import com.sun.net.httpserver.HttpServer;
import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.models.tasks.RunnableTaskException;
import io.kestra.core.runners.RunContext;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.serializers.JacksonMapper;
import io.kestra.core.utils.TestsUtils;
import io.kestra.plugin.scripts.runner.docker.Docker;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@KestraTest
@EnabledIfEnvironmentVariable(named = "DSPY_DOCKER_TESTS", matches = "true")
class DspyDockerTest {
    @Inject
    RunContextFactory runContextFactory;

    @TempDir
    Path directory;

    @Test
    void optimizeThenPredictWithStoredFilesAndExplicitTeacher() throws Exception {
        roundTrip(Map.of(), Map.of());
    }

    @Test
    void optimizeThenPredictWithConfiguredNestedPaths() throws Exception {
        roundTrip(
            Map.of("train", "inputs/examples.jsonl", "validation", "inputs/held out.jsonl",
                "candidate", "compiled/program.json", "result", "reports/evaluation.json"),
            Map.of("dataset", "requests/batch.jsonl", "artifact", "saved/program.json",
                "predictions", "answers/predictions.jsonl", "summary", "reports/summary.json")
        );
    }

    private void roundTrip(Map<String, String> optimizePaths, Map<String, String> predictPaths) throws Exception {
        List<String> models = new CopyOnWriteArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/chat/completions", exchange -> {
            var request = JacksonMapper.ofJson().readTree(exchange.getRequestBody());
            models.add(request.get("model").asText());
            byte[] response = """
                {"id":"fake","object":"chat.completion","created":0,"model":"fake",
                "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant",
                "content":"[[ ## category ## ]]\\nbilling\\n\\n[[ ## completed ## ]]"}}],
                "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
                """.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, response.length);
            try (var body = exchange.getResponseBody()) {
                body.write(response);
            }
        });
        server.start();
        try {
            String endpoint = "http://host.docker.internal:" + server.getAddress().getPort() + "/v1";
            var env = Property.ofValue(Map.of(
                "DSPY_STUDENT_API_BASE", endpoint,
                "DSPY_TEACHER_API_BASE", endpoint,
                "OPENAI_API_KEY", "fake-test-key"
            ));
            Optimize optimize = Optimize.builder()
                .id("optimize")
                .type(Optimize.class.getName())
                .taskRunner(runner())
                .studentModel(Property.ofValue("openai/student"))
                .teacherModel(Property.ofValue("openai/teacher"))
                .signature(Property.ofValue("text -> category"))
                .maxBootstrappedDemos(Property.ofValue(1))
                .maxLabeledDemos(Property.ofValue(1))
                .trainsetUri(Property.ofExpression("{{ inputs.train }}"))
                .validationUri(Property.ofExpression("{{ inputs.validation }}"))
                .filePaths(Property.ofValue(optimizePaths))
                .env(env)
                .build();
            RunContext uploadContext = TestsUtils.mockRunContext(runContextFactory, optimize, Map.of());
            var train = upload(uploadContext, "train", row("train", true));
            var validation = upload(uploadContext, "validation", row("validation", true));
            RunContext optimizeContext = TestsUtils.mockRunContext(runContextFactory, optimize, Map.of("train", train, "validation", validation));
            Optimize.Output candidate = optimize.run(optimizeContext);
            assertEquals(List.of("teacher", "student"), models);
            try (var input = optimizeContext.storage().getFile(candidate.getManifestUri())) {
                var manifest = JacksonMapper.ofJson().readTree(input);
                assertEquals(1.0, manifest.get("validation").get("score").asDouble());
                assertEquals("normalized_exact_match_v1", manifest.get("validation").get("metric").asText());
                assertEquals(1, manifest.get("train").get("count").asInt());
                try (var artifact = optimizeContext.storage().getFile(candidate.getCandidateArtifactUri())) {
                    Path file = directory.resolve("candidate.json");
                    Files.copy(artifact, file);
                    assertEquals(DspyDataset.digest(file), manifest.get("artifactSha256").asText());
                    var state = JacksonMapper.ofJson().readTree(file.toFile());
                    assertEquals(state.get("artifactVersion"), manifest.get("artifactVersion"));
                    assertEquals(state.get("program"), manifest.get("program"));
                }
            }

            var dataset = upload(uploadContext, "dataset", row("test", false));
            Predict predict = Predict.builder()
                .id("predict")
                .type(Predict.class.getName())
                .taskRunner(runner())
                .artifactUri(Property.ofValue(candidate.getCandidateArtifactUri().toString()))
                .datasetUri(Property.ofValue(dataset))
                .filePaths(Property.ofValue(predictPaths))
                .env(env)
                .build();
            RunContext predictContext = TestsUtils.mockRunContext(runContextFactory, predict, Map.of());
            Predict.Output output = predict.run(predictContext);
            try (var input = predictContext.storage().getFile(output.getPredictionsUri())) {
                var prediction = JacksonMapper.ofJson().readTree(input);
                assertEquals("billing", prediction.get("outputs").get("category").asText());
                assertEquals("test", prediction.get("id").asText());
                assertEquals("Send my invoice. {{ untouched }}", prediction.get("inputs").get("text").asText());
                assertTrue(prediction.hasNonNull("artifactVersion"));
            }
            assertEquals(List.of("teacher", "student", "student"), models);
        } finally {
            server.stop(0);
        }
    }

    @Test
    void propagatesPythonFailures() throws Exception {
        Predict predict = Predict.builder()
            .id("predict")
            .type(Predict.class.getName())
            .taskRunner(runner())
            .artifactUri(Property.ofExpression("{{ inputs.artifact }}"))
            .datasetUri(Property.ofExpression("{{ inputs.dataset }}"))
            .build();
        var uploadContext = TestsUtils.mockRunContext(runContextFactory, predict, Map.of());
        String artifact = upload(uploadContext, "artifact", "{}");
        String dataset = upload(uploadContext, "dataset", row("test", false));
        var context = TestsUtils.mockRunContext(runContextFactory, predict, Map.of("artifact", artifact, "dataset", dataset));
        assertThrows(RunnableTaskException.class, () -> predict.run(context));
    }

    private Docker runner() {
        return Docker.builder()
            .type(Docker.class.getName())
            .extraHosts(System.getProperty("os.name").startsWith("Linux") ? List.of("host.docker.internal:host-gateway") : List.of())
            .build();
    }

    private String upload(RunContext context, String name, String content) throws Exception {
        Path file = directory.resolve(name);
        Files.writeString(file, content);
        return context.storage().putFile(file.toFile()).toString();
    }

    private String row(String id, boolean supervised) {
        return "{\"schemaVersion\":1,\"id\":\"" + id + "\",\"inputs\":{\"text\":\"Send my invoice. {{ untouched }}\"}"
            + (supervised ? ",\"reference\":{\"category\":\"billing\"}" : "") + "}\n";
    }
}
