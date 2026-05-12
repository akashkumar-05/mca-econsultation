package com.mca.econsult.controller;

import com.mca.econsult.service.UserService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
public class AuthController {

    private final UserService userService;

    public AuthController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/login")
    public String loginPage(@RequestParam(value = "error", required = false) String error,
                            @RequestParam(value = "logout", required = false) String logout,
                            @RequestParam(value = "registered", required = false) String registered,
                            Model model) {
        if (error != null) {
            model.addAttribute("error", "Invalid username or password.");
        }
        if (logout != null) {
            model.addAttribute("message", "You have been logged out successfully.");
        }
        if (registered != null) {
            model.addAttribute("message", "Registration successful! Please login.");
        }
        return "login";
    }

    @GetMapping("/signup")
    public String signupPage() {
        return "signup";
    }

    @PostMapping("/register")
    public String registerUser(@RequestParam String username,
                               @RequestParam String email,
                               @RequestParam String password,
                               RedirectAttributes redirectAttributes) {
        try {
            if (userService.existsByUsername(username)) {
                redirectAttributes.addFlashAttribute("error", "Username already exists.");
                return "redirect:/signup";
            }
            if (userService.existsByEmail(email)) {
                redirectAttributes.addFlashAttribute("error", "Email already exists.");
                return "redirect:/signup";
            }

            userService.registerUser(username, email, password);
            return "redirect:/login?registered=true";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Registration failed: " + e.getMessage());
            return "redirect:/signup";
        }
    }
}
