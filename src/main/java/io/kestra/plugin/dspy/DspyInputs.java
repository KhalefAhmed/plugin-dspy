package io.kestra.plugin.dspy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import io.kestra.core.models.property.URIFetcher;
import io.kestra.core.runners.RunContext;

final class DspyInputs {

    public static final String TRAIN = "train";
    public static final String VALIDATION = "validation";
    public static final String DATASET = "dataset";

    private DspyInputs() {
    }

    static Map<String, String> prepare(RunContext context, Map<String, Object> config, Map<String, String> files, Map<String, String> locations) throws IOException {
        long maxBytes = ((Number) config.getOrDefault("maxDatasetBytes", DspyDataset.DEFAULT_MAX_BYTES)).longValue();
        int maxRows = ((Number) config.getOrDefault("maxDatasetRows", DspyDataset.DEFAULT_MAX_ROWS)).intValue();
        Map<String, String> staged = new LinkedHashMap<>();
        Map<String, Path> paths = stage(context, files, locations, maxBytes);
        if (!paths.containsKey("artifact")) {
            config.put("program", programDescriptor(config, paths));
        }
        if ("optimize".equals(config.get("operation"))) {
            prepareOptimization(config, paths, locations, maxBytes, maxRows);
        } else {
            preparePrediction(config, paths, locations, maxBytes, maxRows);
        }
        for (var entry : paths.entrySet()) {
            staged.put(locations.get(entry.getKey()), context.storage().putFile(entry.getValue().toFile()).toString());
        }
        return staged;
    }

    private static Map<String, Path> stage(RunContext context, Map<String, String> files, Map<String, String> locations, long maxBytes) throws IOException {
        Map<String, Path> paths = new LinkedHashMap<>();
        for (var entry : files.entrySet()) {
            Path path = context.workingDir().createTempFile();
            try (var stream = URIFetcher.of(entry.getValue()).fetch(context)) {
                if (DATASET.equals(entry.getKey()) || TRAIN.equals(entry.getKey()) || VALIDATION.equals(entry.getKey())) {
                    DspyDataset.copy(stream, path, locations.get(entry.getKey()), maxBytes);
                } else {
                    Files.copy(stream, path, StandardCopyOption.REPLACE_EXISTING);
                }
            }
            paths.put(entry.getKey(), path);
        }
        return paths;
    }

    private static void preparePrediction(Map<String, Object> config, Map<String, Path> paths, Map<String, String> locations, long maxBytes, int maxRows) throws IOException {
        boolean evaluate = Boolean.TRUE.equals(config.get("evaluate"));
        DspyDataset dataset = DspyDataset.read(paths.get(DATASET), locations.get(DATASET), evaluate, maxBytes, maxRows);
        config.put(DATASET, dataset.metadata());
        if (evaluate) {
            referenceFields(config, dataset);
        }
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

    private static void prepareOptimization(Map<String, Object> config, Map<String, Path> paths, Map<String, String> locations, long maxBytes, int maxRows) throws IOException {
        boolean supervised = !config.containsKey("metricFunction");
        DspyDataset train = DspyDataset.read(paths.get(TRAIN), locations.get(TRAIN), supervised, maxBytes, maxRows);
        DspyDataset validation = DspyDataset.read(paths.get(VALIDATION), locations.get(VALIDATION), supervised, maxBytes, maxRows);
        if (train.ids().stream().anyMatch(validation.ids()::contains)) {
            throw new IllegalArgumentException("Training and validation IDs must be disjoint");
        }
        if (supervised) {
            referenceFields(config, train, validation);
        }
        config.put(TRAIN, train.metadata());
        config.put(VALIDATION, validation.metadata());
    }

    private static void referenceFields(Map<String, Object> config, DspyDataset... datasets) {
        var fields = new HashSet<Set<String>>();
        for (DspyDataset dataset : datasets) {
            fields.addAll(dataset.referenceFields());
        }
        if (fields.size() != 1) {
            throw new IllegalArgumentException("All reference objects must use the same output fields");
        }
        config.put("referenceFields", fields.iterator().next());
    }
}
