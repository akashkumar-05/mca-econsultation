package com.mca.econsult.controller;

import com.mca.econsult.model.AnalysisHistory;
import com.mca.econsult.repository.HistoryRepository;
import com.mca.econsult.service.*;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

@Controller
public class AnalysisController {

    private final SentimentService sentimentService;
    private final SummaryService summaryService;
    private final ReportService reportService;
    private final HistoryRepository historyRepository;

    @Value("${file.upload-dir}")
    private String uploadDir;

    @Value("${report.output-dir}")
    private String outputDir;

    public AnalysisController(SentimentService sentimentService,
                               SummaryService summaryService,
                               ReportService reportService,
                               HistoryRepository historyRepository) {
        this.sentimentService = sentimentService;
        this.summaryService = summaryService;
        this.reportService = reportService;
        this.historyRepository = historyRepository;
    }

    @GetMapping("/")
    public String indexPage(Model model, Authentication authentication) {
        model.addAttribute("isLoggedIn", authentication != null && authentication.isAuthenticated()
                && !"anonymousUser".equals(authentication.getPrincipal()));
        if (authentication != null && !"anonymousUser".equals(authentication.getPrincipal())) {
            model.addAttribute("username", authentication.getName());
        }
        return "index";
    }

    @PostMapping("/upload")
    public String uploadFile(@RequestParam("file") MultipartFile file,
                             Authentication authentication,
                             RedirectAttributes redirectAttributes) {
        try {
            // Validate file type
            String filename = file.getOriginalFilename();
            if (filename == null || !filename.toLowerCase().endsWith(".csv")) {
                redirectAttributes.addFlashAttribute("error", "Please upload a CSV file.");
                return "redirect:/";
            }

            // Save uploaded file
            Path uploadPath = Paths.get(uploadDir);
            Files.createDirectories(uploadPath);
            Path filePath = uploadPath.resolve(filename);
            Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);

            // Parse CSV and extract comment_text column
            List<String> comments = parseCSV(filePath.toString());
            if (comments == null || comments.isEmpty()) {
                redirectAttributes.addFlashAttribute("error",
                        "CSV file must contain a 'comment_text' column with data.");
                return "redirect:/";
            }

            // Run sentiment analysis
            List<Map<String, Object>> predictions = sentimentService.predict(comments);

            // Calculate statistics
            int total = predictions.size();
            long positiveCount = predictions.stream()
                    .filter(p -> "positive".equals(p.get("label"))).count();
            long neutralCount = predictions.stream()
                    .filter(p -> "neutral".equals(p.get("label"))).count();
            long negativeCount = predictions.stream()
                    .filter(p -> "negative".equals(p.get("label"))).count();
            long uncertainCount = predictions.stream()
                    .filter(p -> {
                        Object conf = p.get("confidence");
                        double confidence = conf instanceof Number ? ((Number) conf).doubleValue() : 0.0;
                        return confidence < 0.6;
                    }).count();

            double positivePct = total > 0 ? (positiveCount * 100.0 / total) : 0;
            double neutralPct = total > 0 ? (neutralCount * 100.0 / total) : 0;
            double negativePct = total > 0 ? (negativeCount * 100.0 / total) : 0;

            // Generate summaries by sentiment
            Map<String, String> summaries = summaryService.summarizeBySentiment(comments, predictions);

            // Policy Decision Logic
            String policyStatus;
            String statusColor;
            if (positivePct > 60) {
                policyStatus = "ACCEPTED (High Approval)";
                statusColor = "#10b981"; // Success Green
            } else if (positivePct > negativePct && negativePct < 30) {
                policyStatus = "CONDITIONALLY ACCEPTED";
                statusColor = "#f59e0b"; // Warning Orange
            } else if (negativePct > positivePct) {
                policyStatus = "REJECTED (High Opposition)";
                statusColor = "#ef4444"; // Error Red
            } else {
                policyStatus = "REVISION REQUIRED (Divided Opinion)";
                statusColor = "#8b5cf6"; // Info Purple
            }

            // Generate PDF report
            String username = authentication.getName();
            reportService.generateReport(username, total, positivePct, neutralPct, negativePct,
                    (int) uncertainCount, summaries, policyStatus);


            // Save to history
            AnalysisHistory history = new AnalysisHistory(
                    username, filename, positivePct, neutralPct, negativePct, (int) uncertainCount);
            historyRepository.save(history);

            // Categorize by topic
            Map<String, List<Map<String, Object>>> topics =
                    sentimentService.categorizeByTopic(comments, predictions);

            // Build combined data for dashboard
            List<Map<String, Object>> allComments = new ArrayList<>();
            for (int i = 0; i < comments.size(); i++) {
                Map<String, Object> entry = new HashMap<>();
                entry.put("text", comments.get(i));
                entry.put("label", predictions.get(i).get("label"));
                entry.put("confidence", predictions.get(i).get("confidence"));
                allComments.add(entry);
            }

            // Uncertain comments
            List<Map<String, Object>> uncertainComments = allComments.stream()
                    .filter(c -> {
                        Object conf = c.get("confidence");
                        double confidence = conf instanceof Number ? ((Number) conf).doubleValue() : 0.0;
                        return confidence < 0.6;
                    })
                    .collect(Collectors.toList());

            // Pass data to dashboard
            redirectAttributes.addFlashAttribute("analysisComplete", true);
            redirectAttributes.addFlashAttribute("totalComments", total);
            redirectAttributes.addFlashAttribute("positiveCount", positiveCount);
            redirectAttributes.addFlashAttribute("neutralCount", neutralCount);
            redirectAttributes.addFlashAttribute("negativeCount", negativeCount);
            redirectAttributes.addFlashAttribute("positivePct", String.format("%.1f", positivePct));
            redirectAttributes.addFlashAttribute("neutralPct", String.format("%.1f", neutralPct));
            redirectAttributes.addFlashAttribute("negativePct", String.format("%.1f", negativePct));
            redirectAttributes.addFlashAttribute("uncertainCount", uncertainCount);
            redirectAttributes.addFlashAttribute("summaries", summaries);
            redirectAttributes.addFlashAttribute("policyStatus", policyStatus);
            redirectAttributes.addFlashAttribute("statusColor", statusColor);
            redirectAttributes.addFlashAttribute("allComments", allComments);
            redirectAttributes.addFlashAttribute("uncertainComments", uncertainComments);
            redirectAttributes.addFlashAttribute("topics", topics);
            redirectAttributes.addFlashAttribute("filename", filename);

            return "redirect:/dashboard";

        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Analysis failed: " + e.getMessage());
            return "redirect:/";
        } finally {
            // Ensure the uploaded file is deleted to save storage
            try {
                String filename = file.getOriginalFilename();
                if (filename != null) {
                    Path filePath = Paths.get(uploadDir).resolve(filename);
                    Files.deleteIfExists(filePath);
                }
            } catch (IOException e) {
                // Ignore deletion errors
            }
        }
    }

    @GetMapping("/dashboard")
    public String dashboardPage(Model model, Authentication authentication) {
        model.addAttribute("isLoggedIn", authentication != null && authentication.isAuthenticated()
                && !"anonymousUser".equals(authentication.getPrincipal()));
        if (authentication != null && !"anonymousUser".equals(authentication.getPrincipal())) {
            model.addAttribute("username", authentication.getName());
        }
        return "dashboard";
    }


    @GetMapping("/download/pdf")
    public ResponseEntity<Resource> downloadPDF() {
        try {
            File file = new File(outputDir + "report.pdf");
            if (!file.exists()) {
                return ResponseEntity.notFound().build();
            }
            FileSystemResource resource = new FileSystemResource(file);
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=report.pdf")
                    .contentType(MediaType.APPLICATION_PDF)
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Parse CSV file and extract the comment_text column.
     */
    private List<String> parseCSV(String filePath) throws Exception {
        List<String> comments = new ArrayList<>();

        try (CSVReader reader = new CSVReaderBuilder(
                new InputStreamReader(new FileInputStream(filePath), "UTF-8")).build()) {
            String[] header = reader.readNext();
            if (header == null) return null;

            int commentIndex = -1;
            for (int i = 0; i < header.length; i++) {
                // Strip BOM and whitespace before comparing
                String col = header[i].trim().replace("\uFEFF", "");
                if ("comment_text".equalsIgnoreCase(col)) {
                    commentIndex = i;
                    break;
                }
            }

            if (commentIndex == -1) return null;

            String[] line;
            while ((line = reader.readNext()) != null) {
                if (commentIndex < line.length && line[commentIndex] != null
                        && !line[commentIndex].trim().isEmpty()) {
                    comments.add(line[commentIndex].trim());
                }
            }
        }

        return comments;
    }

}
