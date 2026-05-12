package com.mca.econsult.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "analysis_history")
public class AnalysisHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String username;

    @Column(nullable = false)
    private String filename;

    @Column(nullable = false)
    private LocalDateTime timestamp;

    @Column(name = "positive_pct")
    private Double positivePct;

    @Column(name = "neutral_pct")
    private Double neutralPct;

    @Column(name = "negative_pct")
    private Double negativePct;

    @Column(name = "uncertain_count")
    private Integer uncertainCount;

    public AnalysisHistory() {
    }

    public AnalysisHistory(String username, String filename, Double positivePct,
                           Double neutralPct, Double negativePct, Integer uncertainCount) {
        this.username = username;
        this.filename = filename;
        this.positivePct = positivePct;
        this.neutralPct = neutralPct;
        this.negativePct = negativePct;
        this.uncertainCount = uncertainCount;
        this.timestamp = LocalDateTime.now();
    }

    @PrePersist
    protected void onCreate() {
        this.timestamp = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }

    public Double getPositivePct() {
        return positivePct;
    }

    public void setPositivePct(Double positivePct) {
        this.positivePct = positivePct;
    }

    public Double getNeutralPct() {
        return neutralPct;
    }

    public void setNeutralPct(Double neutralPct) {
        this.neutralPct = neutralPct;
    }

    public Double getNegativePct() {
        return negativePct;
    }

    public void setNegativePct(Double negativePct) {
        this.negativePct = negativePct;
    }

    public Integer getUncertainCount() {
        return uncertainCount;
    }

    public void setUncertainCount(Integer uncertainCount) {
        this.uncertainCount = uncertainCount;
    }
}
