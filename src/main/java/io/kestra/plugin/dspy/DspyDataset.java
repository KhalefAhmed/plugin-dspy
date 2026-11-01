package io.kestra.plugin.dspy;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import io.kestra.core.serializers.JacksonMapper;

record DspyDataset(Set<String> ids, Set<Set<String>> referenceFields, String sha256) {
    static final long DEFAULT_MAX_BYTES = 64L * 1024 * 1024;
    static final int DEFAULT_MAX_ROWS = 10_000;

    static DspyDataset read(Path path, boolean supervised) throws IOException {
        return read(path, "dataset.jsonl", supervised, DEFAULT_MAX_BYTES, DEFAULT_MAX_ROWS);
    }

    static DspyDataset read(Path path, String name, boolean supervised, long maxBytes, int maxRows) throws IOException {
        if (Files.size(path) > maxBytes) {
            throw new IllegalArgumentException(name + " exceeds maxDatasetBytes: " + maxBytes);
        }
        Set<String> ids = new HashSet<>();
        Set<Set<String>> referenceFields = new HashSet<>();
        var reader = JacksonMapper.ofJson().reader()
            .with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
        try (var source = Files.newBufferedReader(path)) {
            int lineNumber = 0;
            String line;
            while ((line = source.readLine()) != null) {
                if (lineNumber >= maxRows) {
                    throw new IllegalArgumentException(name + " exceeds maxDatasetRows: " + maxRows);
                }
                String location = name + ":" + ++lineNumber;
                JsonNode row;
                try {
                    row = reader.readTree(line);
                } catch (JsonProcessingException _) {
                    throw new IllegalArgumentException("Invalid JSON at " + location);
                }
                if (!ids.add(identifier(row, location))) {
                    throw new IllegalArgumentException("Duplicate id at " + location);
                }
                referenceFields.add(validatedReferenceFields(row, location, supervised));
                validateNumbers(row, location);
            }
        }
        if (ids.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be empty");
        }
        return new DspyDataset(ids, referenceFields, digest(path));
    }

    static void copy(InputStream source, Path path, String name, long maxBytes) throws IOException {
        try (var target = Files.newOutputStream(path)) {
            byte[] buffer = new byte[8192];
            long total = 0;
            int size;
            while ((size = source.read(buffer)) != -1) {
                if (size > maxBytes - total) {
                    throw new IllegalArgumentException(name + " exceeds maxDatasetBytes: " + maxBytes);
                }
                target.write(buffer, 0, size);
                total += size;
            }
        }
    }

    Map<String, Object> metadata() {
        return Map.of("sha256", sha256, "count", ids.size());
    }

    private static String identifier(JsonNode row, String location) {
        if (row == null || !row.isObject() || !row.path("schemaVersion").isIntegralNumber()
            || !row.path("schemaVersion").bigIntegerValue().equals(java.math.BigInteger.ONE)) {
            throw new IllegalArgumentException(location + " requires schemaVersion: 1");
        }
        JsonNode id = row.path("id");
        if (!id.isTextual() || id.textValue().codePoints().allMatch(
            codePoint -> Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint) || codePoint == 0x85
        )) {
            throw new IllegalArgumentException(location + " id must be a non-empty string");
        }
        return id.textValue();
    }

    private static Set<String> validatedReferenceFields(JsonNode row, String location, boolean supervised) {
        JsonNode inputs = row.path("inputs");
        if (!inputs.isObject()) {
            throw new IllegalArgumentException(location + " requires an inputs object");
        }
        JsonNode reference = row.get("reference");
        if ((reference != null && !reference.isObject()) || (supervised && (reference == null || reference.isEmpty()))) {
            throw new IllegalArgumentException(location + " requires a non-empty reference object");
        }
        Set<String> fields = fields(reference);
        if (fields.stream().anyMatch(inputs::has)) {
            throw new IllegalArgumentException(location + " input and reference fields must be disjoint");
        }
        return fields;
    }

    private static Set<String> fields(JsonNode node) {
        Set<String> fields = new LinkedHashSet<>();
        if (node != null) {
            node.fieldNames().forEachRemaining(fields::add);
        }
        return fields;
    }

    private static void validateNumbers(JsonNode node, String location) {
        if (node.isFloatingPointNumber() && !Double.isFinite(node.doubleValue())) {
            throw new IllegalArgumentException("Non-finite JSON number at " + location);
        }
        for (JsonNode child : node) {
            validateNumbers(child, location);
        }
    }

    static String digest(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (var stream = new DigestInputStream(Files.newInputStream(path), digest)) {
                stream.transferTo(java.io.OutputStream.nullOutputStream());
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }
}
