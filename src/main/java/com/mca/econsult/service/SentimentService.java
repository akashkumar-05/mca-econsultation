package com.mca.econsult.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class SentimentService {

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Sends a list of comment texts to the Python AI service for sentiment prediction.
     * Processes in batches of 32.
     *
     * @param texts list of comment strings
     * @return list of maps with keys: "label" (positive/neutral/negative), "confidence" (double)
     */
    public List<Map<String, Object>> predict(List<String> texts) {
        List<Map<String, Object>> allResults = new ArrayList<>();
        int batchSize = 32;

        for (int i = 0; i < texts.size(); i += batchSize) {
            List<String> batch = texts.subList(i, Math.min(i + batchSize, texts.size()));

            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("texts", batch);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

            try {
                ResponseEntity<String> response = restTemplate.postForEntity(
                        aiServiceUrl + "/predict_legacy", entity, String.class);

                List<Map<String, Object>> batchResults = objectMapper.readValue(
                        response.getBody(), new TypeReference<List<Map<String, Object>>>() {});
                allResults.addAll(batchResults);
            } catch (Exception e) {
                // If AI service fails, mark all as uncertain
                for (int j = 0; j < batch.size(); j++) {
                    Map<String, Object> fallback = new HashMap<>();
                    fallback.put("label", "neutral");
                    fallback.put("confidence", 0.0);
                    allResults.add(fallback);
                }
            }
        }

        return allResults;
    }

    /**
     * Categorize comments into topic buckets based on keywords.
     */
    public Map<String, List<Map<String, Object>>> categorizeByTopic(
            List<String> texts, List<Map<String, Object>> predictions) {

        Map<String, List<String>> topicKeywords = new LinkedHashMap<>();
        topicKeywords.put("Penalties", Arrays.asList("penalty", "penalties", "fine", "fines", "punishment"));
        topicKeywords.put("Audit", Arrays.asList("audit", "auditor", "auditing", "inspection"));
        topicKeywords.put("Reporting", Arrays.asList("report", "reporting", "disclosure", "filing", "annual return"));
        topicKeywords.put("Compliance", Arrays.asList("compliance", "regulation", "regulatory", "rule", "law", "act"));
        topicKeywords.put("Other", Collections.emptyList());

        Map<String, List<Map<String, Object>>> topicResults = new LinkedHashMap<>();
        for (String topic : topicKeywords.keySet()) {
            topicResults.put(topic, new ArrayList<>());
        }

        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i).toLowerCase();
            Map<String, Object> prediction = predictions.get(i);
            boolean categorized = false;

            Map<String, Object> entry = new HashMap<>();
            entry.put("text", texts.get(i));
            entry.put("label", prediction.get("label"));
            entry.put("confidence", prediction.get("confidence"));

            for (Map.Entry<String, List<String>> topicEntry : topicKeywords.entrySet()) {
                if (topicEntry.getKey().equals("Other")) continue;
                for (String keyword : topicEntry.getValue()) {
                    if (text.contains(keyword)) {
                        topicResults.get(topicEntry.getKey()).add(entry);
                        categorized = true;
                        break;
                    }
                }
                if (categorized) break;
            }

            if (!categorized) {
                topicResults.get("Other").add(entry);
            }
        }

        return topicResults;
    }
}
