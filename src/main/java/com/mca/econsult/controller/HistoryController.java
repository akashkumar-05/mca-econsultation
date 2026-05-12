package com.mca.econsult.controller;

import com.mca.econsult.model.AnalysisHistory;
import com.mca.econsult.repository.HistoryRepository;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Controller
public class HistoryController {

    private final HistoryRepository historyRepository;

    public HistoryController(HistoryRepository historyRepository) {
        this.historyRepository = historyRepository;
    }

    @GetMapping("/history")
    public String historyPage(Authentication authentication, Model model) {
        boolean isLoggedIn = authentication != null && authentication.isAuthenticated()
                && !"anonymousUser".equals(authentication.getPrincipal());
        model.addAttribute("isLoggedIn", isLoggedIn);

        if (isLoggedIn) {
            String username = authentication.getName();
            model.addAttribute("username", username);
            List<AnalysisHistory> histories = historyRepository.findByUsernameOrderByTimestampDesc(username);
            model.addAttribute("histories", histories);
        } else {
            // Show all history for anonymous users
            List<AnalysisHistory> histories = historyRepository.findAllByOrderByTimestampDesc();
            model.addAttribute("histories", histories);
        }
        return "history";
    }
}
