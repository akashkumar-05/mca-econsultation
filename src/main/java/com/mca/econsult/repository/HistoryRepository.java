package com.mca.econsult.repository;

import com.mca.econsult.model.AnalysisHistory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface HistoryRepository extends JpaRepository<AnalysisHistory, Long> {

    List<AnalysisHistory> findByUsernameOrderByTimestampDesc(String username);

    List<AnalysisHistory> findAllByOrderByTimestampDesc();
}
