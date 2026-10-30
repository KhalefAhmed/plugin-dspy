package io.kestra.plugin.dspy;

import java.util.Map;

import io.kestra.core.junit.annotations.KestraTest;
import io.kestra.core.models.property.Property;
import io.kestra.core.runners.RunContextFactory;
import io.kestra.core.utils.TestsUtils;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@KestraTest
class DspyTest {
    @Inject
    RunContextFactory runContextFactory;

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
}
