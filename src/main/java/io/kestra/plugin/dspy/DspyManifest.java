package io.kestra.plugin.dspy;

import java.io.IOException;
import java.net.URI;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.kestra.core.runners.RunContext;
import io.kestra.core.serializers.JacksonMapper;

final class DspyManifest {
    private DspyManifest() {
    }

    static URI write(RunContext context, Map<String, Object> config, URI candidateUri, URI resultUri) throws IOException {
        JsonNode candidate = read(context, candidateUri);
        JsonNode result = read(context, resultUri);
        ObjectNode manifest = JacksonMapper.ofJson().createObjectNode();
        manifest.put("schemaVersion", 1);
        manifest.set("artifactVersion", required(candidate, "artifactVersion"));
        manifest.set("program", required(candidate, "program"));
        try (var stream = context.storage().getFile(candidateUri)) {
            var path = context.workingDir().createTempFile();
            java.nio.file.Files.copy(stream, path, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            manifest.put("artifactSha256", DspyDataset.digest(path));
        }
        for (String field : new String[] {"pythonVersion", "dspyVersion", "dependencyVersions", "modelCalls"}) {
            manifest.set(field, required(result, field));
        }
        for (String field : new String[] {
            "studentModel", "teacherModel", "optimizer", "seed", "maxBootstrappedDemos", "maxLabeledDemos",
            "metricThreshold", "maxLlmCalls", "maxTokens", "requestTimeoutSeconds", "train"
        }) {
            manifest.set(field, JacksonMapper.ofJson().valueToTree(config.get(field)));
        }
        ObjectNode validation = JacksonMapper.ofJson().valueToTree(config.get("validation"));
        validation.put("metric", config.containsKey("metricFunction") ? config.get("metricFunction").toString() : "normalized_exact_match_v1");
        validation.set("score", required(result, "validationScore"));
        manifest.set("validation", validation);
        var path = context.workingDir().createTempFile(JacksonMapper.ofJson().writeValueAsBytes(manifest));
        return context.storage().putFile(path.toFile());
    }

    private static JsonNode read(RunContext context, URI uri) throws IOException {
        try (var stream = context.storage().getFile(uri)) {
            return JacksonMapper.ofJson().readTree(stream);
        }
    }

    private static JsonNode required(JsonNode node, String field) {
        if (node == null || !node.hasNonNull(field)) {
            throw new IllegalStateException("DSPy result is missing " + field);
        }
        return node.get(field);
    }
}
