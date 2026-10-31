package io.kestra.plugin.dspy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;

import io.kestra.core.models.property.URIFetcher;
import io.kestra.core.runners.RunContext;

final class DspyInputs {
    private DspyInputs() {
    }

    static Map<String, String> prepare(RunContext context, Map<String, Object> config, Map<String, String> files, Map<String, String> locations) throws IOException {
        Map<String, String> staged = new LinkedHashMap<>();
        Map<String, Path> paths = new LinkedHashMap<>();
        for (var entry : files.entrySet()) {
            Path path = context.workingDir().createTempFile();
            try (var stream = URIFetcher.of(entry.getValue()).fetch(context)) {
                Files.copy(stream, path, StandardCopyOption.REPLACE_EXISTING);
            }
            paths.put(entry.getKey(), path);
        }
        if (!paths.containsKey("artifact")) {
            config.put("program", programDescriptor(config, paths));
        }
        if ("optimize".equals(config.get("operation"))) {
            prepareOptimization(config, paths, locations);
        } else {
            DspyDataset dataset = DspyDataset.read(paths.get("dataset"), locations.get("dataset"), false);
            config.put("datasets", Map.of("dataset", dataset.records()));
        }
        for (var entry : paths.entrySet()) {
            staged.put(locations.get(entry.getKey()), context.storage().putFile(entry.getValue().toFile()).toString());
        }
        return staged;
    }

    private static Map<String, Object> programDescriptor(Map<String, Object> config, Map<String, Path> paths) throws IOException {
        Map<String, Object> program = new LinkedHashMap<>();
        if (config.get("signature") instanceof String signature) {
            if (signature.isBlank()) {
                throw new IllegalArgumentException("signature must not be blank");
            }
            program.put("kind", "declarative");
            program.put("signature", signature);
            program.put("predictor", config.get("predictor"));
            program.put("instructions", config.get("instructions"));
        } else {
            program.put("kind", "python");
            program.put("factory", config.get("programFactory"));
            program.put("sha256", DspyDataset.digest(paths.get("program")));
        }
        return program;
    }

    private static void prepareOptimization(Map<String, Object> config, Map<String, Path> paths, Map<String, String> locations) throws IOException {
        boolean supervised = !config.containsKey("metricFunction");
        DspyDataset train = DspyDataset.read(paths.get("train"), locations.get("train"), supervised);
        DspyDataset validation = DspyDataset.read(paths.get("validation"), locations.get("validation"), supervised);
        if (train.ids().stream().anyMatch(validation.ids()::contains)) {
            throw new IllegalArgumentException("Training and validation IDs must be disjoint");
        }
        if (supervised) {
            var fields = new HashSet<>(train.referenceFields());
            fields.addAll(validation.referenceFields());
            if (fields.size() != 1) {
                throw new IllegalArgumentException("All reference objects must use the same output fields");
            }
            config.put("referenceFields", fields.iterator().next());
        }
        config.put("train", train.metadata());
        config.put("validation", validation.metadata());
        config.put("datasets", Map.of("train", train.records(), "validation", validation.records()));
    }
}
