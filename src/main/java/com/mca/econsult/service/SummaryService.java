package com.mca.econsult.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class SummaryService {

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Sends texts to the Python AI service for summarization.
     * Uses first 20 comments, max 3500 chars.
     *
     * @param texts list of comment strings
     * @return summary string
     */
    public String summarize(List<String> texts) {
        // Limit to first 20 comments
        List<String> limited = texts.subList(0, Math.min(20, texts.size()));

        // Limit total characters to 3500
        List<String> truncated = new ArrayList<>();
        int totalChars = 0;
        for (String text : limited) {
            if (totalChars + text.length() > 3500) {
                int remaining = 3500 - totalChars;
                if (remaining > 0) {
                    truncated.add(text.substring(0, remaining));
                }
                break;
            }
            truncated.add(text);
            totalChars += text.length();
        }

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("texts", truncated);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(
                    aiServiceUrl + "/summarize", entity, String.class);

            Map<String, Object> result = objectMapper.readValue(response.getBody(), Map.class);
            // Handle both new wrapped format {"success":true,"data":{"summary":"..."}}
            // and legacy flat format {"summary":"..."}
            Object dataObj = result.get("data");
            if (dataObj instanceof Map) {
                Map<String, Object> data = (Map<String, Object>) dataObj;
                return (String) data.get("summary");
            }
            return (String) result.get("summary");
        } catch (Exception e) {
            return "Summary generation failed: " + e.getMessage();
        }
    }

    /**
     * Generate summaries for each sentiment group.
     */
    public Map<String, String> summarizeBySentiment(List<String> texts, List<Map<String, Object>> predictions) {
        Map<String, List<String>> grouped = new HashMap<>();
        grouped.put("positive", new ArrayList<>());
        grouped.put("neutral", new ArrayList<>());
        grouped.put("negative", new ArrayList<>());

        for (int i = 0; i < texts.size(); i++) {
            String label = (String) predictions.get(i).get("label");
            grouped.getOrDefault(label, grouped.get("neutral")).add(texts.get(i));
        }

        Map<String, String> summaries = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : grouped.entrySet()) {
            if (!entry.getValue().isEmpty()) {
                summaries.put(entry.getKey(), summarize(entry.getValue()));
            } else {
                summaries.put(entry.getKey(), "No " + entry.getKey() + " comments found.");
            }
        }

        return summaries;
    }
}
