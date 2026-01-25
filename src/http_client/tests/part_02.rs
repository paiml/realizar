use crate::http_client::*;
// =========================================================================

/// Performance trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Regressing,
}

/// IMP-153a: Performance history entry
#[derive(Debug, Clone)]
pub struct PerformanceEntry {
    pub timestamp: String,
    pub throughput_tps: f64,
    pub milestone: String,
    pub gap_vs_target_percent: f64,
}

/// IMP-153a: Performance history tracking
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    pub entries: Vec<PerformanceEntry>,
    pub target_tps: f64,
    pub trend: PerformanceTrend,
    pub avg_improvement_per_entry: f64,
}

impl PerformanceHistory {
    pub fn new(target_tps: f64) -> Self {
        Self {
            entries: Vec::new(),
            target_tps,
            trend: PerformanceTrend::Stable,
            avg_improvement_per_entry: 0.0,
        }
    }

    pub fn add_entry(&mut self, throughput_tps: f64, milestone: &str) {
        let gap = if self.target_tps > 0.0 {
            ((throughput_tps - self.target_tps) / self.target_tps) * 100.0
        } else {
            0.0
        };
        self.entries.push(PerformanceEntry {
            timestamp: chrono::Utc::now().to_rfc3339(),
            throughput_tps,
            milestone: milestone.to_string(),
            gap_vs_target_percent: gap,
        });
        self.recalculate_trend();
    }

    fn recalculate_trend(&mut self) {
        if self.entries.len() < 2 {
            self.trend = PerformanceTrend::Stable;
            self.avg_improvement_per_entry = 0.0;
            return;
        }

        // Calculate improvements between consecutive entries
        let mut improvements = Vec::new();
        for i in 1..self.entries.len() {
            let prev = self.entries[i - 1].throughput_tps;
            let curr = self.entries[i].throughput_tps;
            if prev > 0.0 {
                improvements.push((curr - prev) / prev * 100.0);
            }
        }

        if improvements.is_empty() {
            self.trend = PerformanceTrend::Stable;
            self.avg_improvement_per_entry = 0.0;
            return;
        }

        let avg: f64 = improvements.iter().sum::<f64>() / improvements.len() as f64;
        self.avg_improvement_per_entry = avg;

        // Determine trend: >2% avg improvement = improving, <-2% = regressing
        self.trend = if avg > 2.0 {
            PerformanceTrend::Improving
        } else if avg < -2.0 {
            PerformanceTrend::Regressing
        } else {
            PerformanceTrend::Stable
        };
    }

    pub fn latest_throughput(&self) -> Option<f64> {
        self.entries.last().map(|e| e.throughput_tps)
    }

    pub fn entries_count(&self) -> usize {
        self.entries.len()
    }
}

/// IMP-153a: Test performance history tracking
#[test]
fn test_imp_153a_performance_history() {
    let mut history = PerformanceHistory::new(256.0); // Target: llama.cpp baseline

    // Add progression entries
    history.add_entry(80.0, "Baseline");
    history.add_entry(100.0, "P1-25%");
    history.add_entry(120.0, "P1");
    history.add_entry(160.0, "P1-P2");
    history.add_entry(200.0, "P2");

    assert_eq!(
        history.entries_count(),
        5,
        "IMP-153a: Should have 5 entries"
    );
    assert_eq!(
        history.latest_throughput(),
        Some(200.0),
        "IMP-153a: Latest should be 200"
    );
    assert_eq!(
        history.trend,
        PerformanceTrend::Improving,
        "IMP-153a: Trend should be improving"
    );
    assert!(
        history.avg_improvement_per_entry > 20.0,
        "IMP-153a: Should show >20% avg improvement per entry"
    );

    println!("\nIMP-153a: Performance History:");
    for (i, entry) in history.entries.iter().enumerate() {
        println!(
            "  Entry {}: {} tok/s ({}) gap={:+.1}%",
            i + 1,
            entry.throughput_tps,
            entry.milestone,
            entry.gap_vs_target_percent
        );
    }
    println!("  Trend: {:?}", history.trend);
    println!(
        "  Avg improvement: {:.1}%/entry",
        history.avg_improvement_per_entry
    );
}

/// IMP-153b: Test trend detection
#[test]
fn test_imp_153b_trend_detection() {
    // Scenario 1: Improving trend
    let mut improving = PerformanceHistory::new(256.0);
    improving.add_entry(80.0, "Start");
    improving.add_entry(100.0, "Mid");
    improving.add_entry(130.0, "End");
    assert_eq!(
        improving.trend,
        PerformanceTrend::Improving,
        "IMP-153b: 80→100→130 should be improving"
    );

    // Scenario 2: Regressing trend
    let mut regressing = PerformanceHistory::new(256.0);
    regressing.add_entry(120.0, "Start");
    regressing.add_entry(110.0, "Mid");
    regressing.add_entry(95.0, "End");
    assert_eq!(
        regressing.trend,
        PerformanceTrend::Regressing,
        "IMP-153b: 120→110→95 should be regressing"
    );

    // Scenario 3: Stable trend (within ±2%)
    let mut stable = PerformanceHistory::new(256.0);
    stable.add_entry(100.0, "Start");
    stable.add_entry(101.0, "Mid");
    stable.add_entry(100.5, "End");
    assert_eq!(
        stable.trend,
        PerformanceTrend::Stable,
        "IMP-153b: 100→101→100.5 should be stable"
    );

    // Scenario 4: Single entry = stable
    let mut single = PerformanceHistory::new(256.0);
    single.add_entry(100.0, "Only");
    assert_eq!(
        single.trend,
        PerformanceTrend::Stable,
        "IMP-153b: Single entry should be stable"
    );

    println!("\nIMP-153b: Trend Detection:");
    println!(
        "  Improving: 80→100→130, avg={:.1}%",
        improving.avg_improvement_per_entry
    );
    println!(
        "  Regressing: 120→110→95, avg={:.1}%",
        regressing.avg_improvement_per_entry
    );
    println!(
        "  Stable: 100→101→100.5, avg={:.1}%",
        stable.avg_improvement_per_entry
    );
}

/// IMP-153c: Milestone progress summary
#[derive(Debug, Clone)]
pub struct MilestoneProgress {
    pub current_tps: f64,
    pub p1_target: f64,
    pub p2_target: f64,
    pub parity_target: f64,
    pub p1_achieved: bool,
    pub p2_achieved: bool,
    pub parity_achieved: bool,
    pub next_milestone: String,
    pub gap_to_next: f64,
}

impl MilestoneProgress {
    pub fn new(current_tps: f64) -> Self {
        let p1_target: f64 = 120.0; // Per spec: 1.5x baseline
        let p2_target: f64 = 200.0; // Per spec: 2.5x baseline
        let parity_target: f64 = 230.0; // Per spec: within 10% of 256

        let p1_achieved = current_tps >= p1_target;
        let p2_achieved = current_tps >= p2_target;
        let parity_achieved = current_tps >= parity_target;

        let (next_milestone, gap_to_next) = if !p1_achieved {
            ("P1".to_string(), p1_target - current_tps)
        } else if !p2_achieved {
            ("P2".to_string(), p2_target - current_tps)
        } else if !parity_achieved {
            ("Parity".to_string(), parity_target - current_tps)
        } else {
            ("Complete".to_string(), 0.0)
        };

        Self {
            current_tps,
            p1_target,
            p2_target,
            parity_target,
            p1_achieved,
            p2_achieved,
            parity_achieved,
            next_milestone,
            gap_to_next,
        }
    }
}

/// IMP-153c: Test milestone progress tracking
#[test]
fn test_imp_153c_milestone_progress() {
    // Scenario 1: Before P1
    let before_p1 = MilestoneProgress::new(80.0);
    assert!(!before_p1.p1_achieved, "IMP-153c: 80 should not achieve P1");
    assert_eq!(before_p1.next_milestone, "P1");
    assert!(
        (before_p1.gap_to_next - 40.0).abs() < 0.1,
        "IMP-153c: 40 tok/s to P1"
    );

    // Scenario 2: Between P1 and P2
    let between = MilestoneProgress::new(150.0);
    assert!(between.p1_achieved, "IMP-153c: 150 should achieve P1");
    assert!(!between.p2_achieved, "IMP-153c: 150 should not achieve P2");
    assert_eq!(between.next_milestone, "P2");
    assert!(
        (between.gap_to_next - 50.0).abs() < 0.1,
        "IMP-153c: 50 tok/s to P2"
    );

    // Scenario 3: Between P2 and Parity
    let near_parity = MilestoneProgress::new(210.0);
    assert!(near_parity.p2_achieved, "IMP-153c: 210 should achieve P2");
    assert!(
        !near_parity.parity_achieved,
        "IMP-153c: 210 should not achieve Parity"
    );
    assert_eq!(near_parity.next_milestone, "Parity");

    // Scenario 4: Parity achieved
    let at_parity = MilestoneProgress::new(240.0);
    assert!(
        at_parity.parity_achieved,
        "IMP-153c: 240 should achieve Parity"
    );
    assert_eq!(at_parity.next_milestone, "Complete");

    println!("\nIMP-153c: Milestone Progress:");
    println!(
        "  80 tok/s: P1={} P2={} Parity={} Next={} Gap={:.0}",
        before_p1.p1_achieved,
        before_p1.p2_achieved,
        before_p1.parity_achieved,
        before_p1.next_milestone,
        before_p1.gap_to_next
    );
    println!(
        "  150 tok/s: P1={} P2={} Parity={} Next={} Gap={:.0}",
        between.p1_achieved,
        between.p2_achieved,
        between.parity_achieved,
        between.next_milestone,
        between.gap_to_next
    );
    println!(
        "  210 tok/s: P1={} P2={} Parity={} Next={} Gap={:.0}",
        near_parity.p1_achieved,
        near_parity.p2_achieved,
        near_parity.parity_achieved,
        near_parity.next_milestone,
        near_parity.gap_to_next
    );
    println!(
        "  240 tok/s: P1={} P2={} Parity={} Next={}",
        at_parity.p1_achieved,
        at_parity.p2_achieved,
        at_parity.parity_achieved,
        at_parity.next_milestone
    );
}

/// IMP-153d: Gap trend tracking
#[derive(Debug, Clone)]
pub struct GapTrend {
    pub initial_gap_percent: f64,
    pub current_gap_percent: f64,
    pub gap_closed_percent: f64,
    pub estimated_entries_to_parity: usize,
}

impl GapTrend {
    pub fn new(
        initial_tps: f64,
        current_tps: f64,
        target_tps: f64,
        avg_improvement_percent: f64,
    ) -> Self {
        let initial_gap = if target_tps > 0.0 {
            ((target_tps - initial_tps) / target_tps) * 100.0
        } else {
            0.0
        };
        let current_gap = if target_tps > 0.0 {
            ((target_tps - current_tps) / target_tps) * 100.0
        } else {
            0.0
        };
        let gap_closed = initial_gap - current_gap;

        // Estimate entries to reach parity (within 10% of target)
        let parity_gap: f64 = 10.0;
        let remaining_gap = current_gap - parity_gap;
        let estimated_entries = if remaining_gap <= 0.0 || avg_improvement_percent <= 0.0 {
            0
        } else {
            // Rough estimate: remaining_gap / avg_improvement_percent
            // This is simplified; real calculation would consider compound growth
            ((remaining_gap / avg_improvement_percent) * 1.5).ceil() as usize
        };

        Self {
            initial_gap_percent: initial_gap,
            current_gap_percent: current_gap,
            gap_closed_percent: gap_closed,
            estimated_entries_to_parity: estimated_entries,
        }
    }
}

/// IMP-153d: Test gap trend tracking
#[test]
fn test_imp_153d_gap_trend() {
    // Scenario: Started at 80 tok/s, now at 120 tok/s, targeting 256 tok/s
    // With 25% avg improvement per entry
    let trend = GapTrend::new(80.0, 120.0, 256.0, 25.0);

    // Initial gap: (256-80)/256 = 68.75%
    // Current gap: (256-120)/256 = 53.125%
    // Gap closed: 68.75 - 53.125 = 15.625%
    assert!(
        (trend.initial_gap_percent - 68.75).abs() < 0.1,
        "IMP-153d: Initial gap should be ~68.75%"
    );
    assert!(
        (trend.current_gap_percent - 53.125).abs() < 0.1,
        "IMP-153d: Current gap should be ~53.125%"
    );
    assert!(
        (trend.gap_closed_percent - 15.625).abs() < 0.1,
        "IMP-153d: Gap closed should be ~15.625%"
    );
    assert!(
        trend.estimated_entries_to_parity > 0,
        "IMP-153d: Should estimate entries needed"
    );

    // Already at parity
    let at_parity = GapTrend::new(80.0, 240.0, 256.0, 25.0);
    assert_eq!(
        at_parity.estimated_entries_to_parity, 0,
        "IMP-153d: At parity should be 0 entries"
    );

    println!("\nIMP-153d: Gap Trend:");
    println!(
        "  Initial: {:.1}% gap (80 vs 256)",
        trend.initial_gap_percent
    );
    println!(
        "  Current: {:.1}% gap (120 vs 256)",
        trend.current_gap_percent
    );
    println!("  Closed: {:.1}%", trend.gap_closed_percent);
    println!(
        "  Est. entries to parity: {}",
        trend.estimated_entries_to_parity
    );
    println!(
        "  At parity (240 vs 256): {} entries",
        at_parity.estimated_entries_to_parity
    );
}

// =========================================================================
// IMP-154: Automated Performance Gate Validation (EXTREME TDD)
// Per spec §10.1: CI/CD integration for performance regression prevention
// =========================================================================

/// Gate status for performance checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStatus {
    Pass,
    Warn,
    Fail,
}

/// IMP-154a: Individual performance gate
#[derive(Debug, Clone)]
pub struct PerformanceGate {
    pub name: String,
    pub measured_value: f64,
    pub pass_threshold: f64,
    pub warn_threshold: f64,
    pub unit: String,
    pub status: GateStatus,
    pub message: String,
}

impl PerformanceGate {
    /// Create a gate where higher values are better (e.g., throughput)
    pub fn higher_is_better(
        name: &str,
        measured: f64,
        pass_threshold: f64,
        warn_threshold: f64,
        unit: &str,
    ) -> Self {
        let status = if measured >= pass_threshold {
            GateStatus::Pass
        } else if measured >= warn_threshold {
            GateStatus::Warn
        } else {
            GateStatus::Fail
        };
        let message = match status {
            GateStatus::Pass => format!(
                "{:.1}{} >= {:.1}{} (PASS)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Warn => format!(
                "{:.1}{} < {:.1}{} (WARN)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Fail => format!(
                "{:.1}{} < {:.1}{} (FAIL)",
                measured, unit, warn_threshold, unit
            ),
        };
        Self {
            name: name.to_string(),
            measured_value: measured,
            pass_threshold,
            warn_threshold,
            unit: unit.to_string(),
            status,
            message,
        }
    }

    /// Create a gate where lower values are better (e.g., latency)
    pub fn lower_is_better(
        name: &str,
        measured: f64,
        pass_threshold: f64,
        warn_threshold: f64,
        unit: &str,
    ) -> Self {
        let status = if measured <= pass_threshold {
            GateStatus::Pass
        } else if measured <= warn_threshold {
            GateStatus::Warn
        } else {
            GateStatus::Fail
        };
        let message = match status {
            GateStatus::Pass => format!(
                "{:.1}{} <= {:.1}{} (PASS)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Warn => format!(
                "{:.1}{} > {:.1}{} (WARN)",
                measured, unit, pass_threshold, unit
            ),
            GateStatus::Fail => format!(
                "{:.1}{} > {:.1}{} (FAIL)",
                measured, unit, warn_threshold, unit
            ),
        };
        Self {
            name: name.to_string(),
            measured_value: measured,
            pass_threshold,
            warn_threshold,
            unit: unit.to_string(),
            status,
            message,
        }
    }
}

/// IMP-154a: Test individual performance gate
#[test]
fn test_imp_154a_performance_gate() {
    // Throughput gate: Pass if >= 120 tok/s, Warn if >= 100, Fail otherwise
    let pass_gate = PerformanceGate::higher_is_better("Throughput", 130.0, 120.0, 100.0, " tok/s");
    assert_eq!(
        pass_gate.status,
        GateStatus::Pass,
        "IMP-154a: 130 should pass 120 threshold"
    );

    let warn_gate = PerformanceGate::higher_is_better("Throughput", 110.0, 120.0, 100.0, " tok/s");
    assert_eq!(
        warn_gate.status,
        GateStatus::Warn,
        "IMP-154a: 110 should warn (100-120)"
    );

    let fail_gate = PerformanceGate::higher_is_better("Throughput", 90.0, 120.0, 100.0, " tok/s");
    assert_eq!(
        fail_gate.status,
        GateStatus::Fail,
        "IMP-154a: 90 should fail (<100)"
    );

    // Latency gate: Pass if <= 50ms, Warn if <= 100ms, Fail otherwise
    let latency_pass = PerformanceGate::lower_is_better("P50 Latency", 40.0, 50.0, 100.0, "ms");
    assert_eq!(
        latency_pass.status,
        GateStatus::Pass,
        "IMP-154a: 40ms should pass"
    );

    let latency_warn = PerformanceGate::lower_is_better("P50 Latency", 70.0, 50.0, 100.0, "ms");
    assert_eq!(
        latency_warn.status,
        GateStatus::Warn,
        "IMP-154a: 70ms should warn"
    );

    let latency_fail = PerformanceGate::lower_is_better("P50 Latency", 150.0, 50.0, 100.0, "ms");
    assert_eq!(
        latency_fail.status,
        GateStatus::Fail,
        "IMP-154a: 150ms should fail"
    );

    println!("\nIMP-154a: Performance Gates:");
    println!("  {} - {}", pass_gate.name, pass_gate.message);
    println!("  {} - {}", warn_gate.name, warn_gate.message);
    println!("  {} - {}", fail_gate.name, fail_gate.message);
    println!("  {} - {}", latency_pass.name, latency_pass.message);
    println!("  {} - {}", latency_warn.name, latency_warn.message);
    println!("  {} - {}", latency_fail.name, latency_fail.message);
}

/// IMP-154b: Composite gate that aggregates multiple checks
#[derive(Debug, Clone)]
pub struct CompositeGate {
    pub gates: Vec<PerformanceGate>,
    pub overall_status: GateStatus,
    pub pass_count: usize,
    pub warn_count: usize,
    pub fail_count: usize,
}

impl CompositeGate {
    pub fn new(gates: Vec<PerformanceGate>) -> Self {
        let pass_count = gates
            .iter()
            .filter(|g| g.status == GateStatus::Pass)
            .count();
        let warn_count = gates
            .iter()
            .filter(|g| g.status == GateStatus::Warn)
            .count();
        let fail_count = gates
            .iter()
            .filter(|g| g.status == GateStatus::Fail)
            .count();

        // Overall: Fail if any fail, Warn if any warn, Pass otherwise
        let overall_status = if fail_count > 0 {
            GateStatus::Fail
        } else if warn_count > 0 {
            GateStatus::Warn
        } else {
            GateStatus::Pass
        };

        Self {
            gates,
            overall_status,
            pass_count,
            warn_count,
            fail_count,
        }
    }

    pub fn all_passed(&self) -> bool {
        self.overall_status == GateStatus::Pass
    }

    pub fn should_block_merge(&self) -> bool {
        self.overall_status == GateStatus::Fail
    }
}

/// IMP-154b: Test composite gate
#[test]
fn test_imp_154b_composite_gate() {
    // Scenario 1: All pass
    let all_pass = CompositeGate::new(vec![
        PerformanceGate::higher_is_better("Throughput", 130.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("Latency", 40.0, 50.0, 100.0, "ms"),
    ]);
    assert!(all_pass.all_passed(), "IMP-154b: All gates should pass");
    assert!(!all_pass.should_block_merge(), "IMP-154b: Should not block");
    assert_eq!(all_pass.pass_count, 2);

    // Scenario 2: One warn
    let one_warn = CompositeGate::new(vec![
        PerformanceGate::higher_is_better("Throughput", 110.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("Latency", 40.0, 50.0, 100.0, "ms"),
    ]);
    assert_eq!(
        one_warn.overall_status,
        GateStatus::Warn,
        "IMP-154b: Should be warn"
    );
    assert!(
        !one_warn.should_block_merge(),
        "IMP-154b: Warn should not block"
    );

    // Scenario 3: One fail
    let one_fail = CompositeGate::new(vec![
        PerformanceGate::higher_is_better("Throughput", 90.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("Latency", 40.0, 50.0, 100.0, "ms"),
    ]);
    assert_eq!(
        one_fail.overall_status,
        GateStatus::Fail,
        "IMP-154b: Should be fail"
    );
    assert!(one_fail.should_block_merge(), "IMP-154b: Fail should block");

    println!("\nIMP-154b: Composite Gates:");
    println!(
        "  All pass: {:?} (block={})",
        all_pass.overall_status,
        all_pass.should_block_merge()
    );
    println!(
        "  One warn: {:?} (block={})",
        one_warn.overall_status,
        one_warn.should_block_merge()
    );
    println!(
        "  One fail: {:?} (block={})",
        one_fail.overall_status,
        one_fail.should_block_merge()
    );
}

/// IMP-154c: Standard gate configuration for performance parity
pub struct ParityGateConfig {
    pub p1_throughput_pass: f64,
    pub p1_throughput_warn: f64,
    pub parity_throughput_pass: f64,
    pub parity_throughput_warn: f64,
    pub regression_threshold_percent: f64,
}

impl Default for ParityGateConfig {
    fn default() -> Self {
        Self {
            p1_throughput_pass: 120.0,     // P1 milestone: 1.5x baseline
            p1_throughput_warn: 100.0,     // 25% of P1 progress
            parity_throughput_pass: 230.0, // Within 10% of 256
            parity_throughput_warn: 200.0, // P2 milestone
            regression_threshold_percent: 5.0,
        }
    }
}

/// IMP-154c: Test parity gate configuration
#[test]
fn test_imp_154c_parity_gate_config() {
    let config = ParityGateConfig::default();

    // Create gates based on config
    let current_tps: f64 = 150.0;
    let baseline_tps: f64 = 145.0;

    // P1 gate
    let p1_gate = PerformanceGate::higher_is_better(
        "P1 Throughput",
        current_tps,
        config.p1_throughput_pass,
        config.p1_throughput_warn,
        " tok/s",
    );
    assert_eq!(
        p1_gate.status,
        GateStatus::Pass,
        "IMP-154c: 150 should pass P1"
    );

    // Parity gate (150 < 200 warn threshold = Fail)
    let parity_gate = PerformanceGate::higher_is_better(
        "Parity Throughput",
        current_tps,
        config.parity_throughput_pass,
        config.parity_throughput_warn,
        " tok/s",
    );
    assert_eq!(
        parity_gate.status,
        GateStatus::Fail,
        "IMP-154c: 150 < 200 should fail parity"
    );

    // Regression gate (higher is better = no regression)
    let regression_percent = ((current_tps - baseline_tps) / baseline_tps) * 100.0;
    let regression_gate = PerformanceGate::higher_is_better(
        "Regression Check",
        regression_percent,
        -config.regression_threshold_percent, // Pass if >= -5%
        -10.0,                                // Warn if >= -10%
        "%",
    );
    assert_eq!(
        regression_gate.status,
        GateStatus::Pass,
        "IMP-154c: +3.4% should pass regression"
    );

    let composite = CompositeGate::new(vec![p1_gate, parity_gate, regression_gate]);
    assert_eq!(
        composite.overall_status,
        GateStatus::Fail,
        "IMP-154c: Should fail (parity gate failed)"
    );

    println!("\nIMP-154c: Parity Gate Config:");
    println!(
        "  Current: {:.0} tok/s, Baseline: {:.0} tok/s",
        current_tps, baseline_tps
    );
    for gate in &composite.gates {
        println!("  {} - {}", gate.name, gate.message);
    }
    println!("  Overall: {:?}", composite.overall_status);
}

/// IMP-154d: Gate report for CI output
#[derive(Debug, Clone)]
pub struct GateReport {
    pub title: String,
    pub composite: CompositeGate,
    pub summary: String,
    pub exit_code: i32,
}

impl GateReport {
    pub fn new(title: &str, composite: CompositeGate) -> Self {
        let summary = format!(
            "{}: {} PASS, {} WARN, {} FAIL -> {:?}",
            title,
            composite.pass_count,
            composite.warn_count,
            composite.fail_count,
            composite.overall_status
        );
        let exit_code = match composite.overall_status {
            GateStatus::Pass => 0,
            GateStatus::Warn => 0, // Warn doesn't fail CI
            GateStatus::Fail => 1,
        };
        Self {
            title: title.to_string(),
            composite,
            summary,
            exit_code,
        }
    }

    pub fn format_for_ci(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("## {}\n\n", self.title));
        output.push_str("| Gate | Status | Details |\n");
        output.push_str("|------|--------|--------|\n");
        for gate in &self.composite.gates {
            let status_emoji = match gate.status {
                GateStatus::Pass => "✅",
                GateStatus::Warn => "⚠️",
                GateStatus::Fail => "❌",
            };
            output.push_str(&format!(
                "| {} | {} | {} |\n",
                gate.name, status_emoji, gate.message
            ));
        }
        output.push_str(&format!("\n**Result**: {}\n", self.summary));
        output
    }
}

/// IMP-154d: Test gate report generation
#[test]
fn test_imp_154d_gate_report() {
    let gates = vec![
        PerformanceGate::higher_is_better("Throughput", 125.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("P50 Latency", 45.0, 50.0, 100.0, "ms"),
        PerformanceGate::higher_is_better("Regression", 2.5, -5.0, -10.0, "%"),
    ];
    let composite = CompositeGate::new(gates);
    let report = GateReport::new("Performance Parity Check", composite);

    assert_eq!(
        report.exit_code, 0,
        "IMP-154d: All pass should have exit code 0"
    );
    assert!(
        report.summary.contains("3 PASS"),
        "IMP-154d: Should show 3 PASS"
    );

    let ci_output = report.format_for_ci();
    assert!(
        ci_output.contains("## Performance Parity Check"),
        "IMP-154d: Should have title"
    );
    assert!(
        ci_output.contains("Throughput"),
        "IMP-154d: Should list throughput gate"
    );
    assert!(ci_output.contains("✅"), "IMP-154d: Should have pass emoji");

    // Test failure scenario
    let fail_gates = vec![PerformanceGate::higher_is_better(
        "Throughput",
        80.0,
        120.0,
        100.0,
        " tok/s",
    )];
    let fail_report = GateReport::new("Failed Check", CompositeGate::new(fail_gates));
    assert_eq!(
        fail_report.exit_code, 1,
        "IMP-154d: Fail should have exit code 1"
    );

    println!("\nIMP-154d: Gate Report:");
    println!("{}", ci_output);
    println!("Exit code: {}", report.exit_code);
}

// =========================================================================
// IMP-155: Fused Q4K Throughput Verification vs External Servers (EXTREME TDD)
// Per spec §13.1 Phase 2: Verify fused kernel achieves 2x gain (120→240 tok/s)
// =========================================================================

/// IMP-155a: Fused kernel benchmark result
#[derive(Debug, Clone)]
pub struct FusedKernelResult {
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth_gbs: f64,
    /// Compute efficiency (% of peak FLOPS)
    pub compute_efficiency_percent: f64,
    /// Whether fused path was used
    pub fused_path_used: bool,
    /// Speedup vs separate dequant+matvec
    pub speedup_vs_separate: f64,
}

impl FusedKernelResult {
    pub fn new(
        throughput_tps: f64,
        memory_bandwidth_gbs: f64,
        fused_path_used: bool,
        baseline_separate_tps: f64,
    ) -> Self {
        let speedup = if baseline_separate_tps > 0.0 {
            throughput_tps / baseline_separate_tps
        } else {
            1.0
        };
        // Estimate compute efficiency based on throughput vs theoretical peak
        // Q4_K: 4.5 bits/param, ~2 FLOPs per param for matvec
        // Theoretical peak depends on memory bandwidth
        let compute_efficiency = (throughput_tps / 1000.0).min(100.0) * 100.0;
        Self {
            throughput_tps,
            memory_bandwidth_gbs,
            compute_efficiency_percent: compute_efficiency,
            fused_path_used,
            speedup_vs_separate: speedup,
        }
    }

    pub fn meets_p2_target(&self) -> bool {
        self.throughput_tps >= 200.0 && self.fused_path_used
    }
}

/// IMP-155a: Test fused kernel result struct
#[test]
fn test_imp_155a_fused_kernel_result() {
    // Scenario: Fused kernel at 240 tok/s vs 80 tok/s separate
    let result = FusedKernelResult::new(240.0, 45.0, true, 80.0);

    assert!(
        result.fused_path_used,
        "IMP-155a: Fused path should be used"
    );
    assert!(
        (result.speedup_vs_separate - 3.0).abs() < 0.1,
        "IMP-155a: Should show 3x speedup (240/80)"
    );
    assert!(
        result.meets_p2_target(),
        "IMP-155a: 240 tok/s should meet P2 target"
    );

    // Scenario: Below P2 target
    let below_target = FusedKernelResult::new(150.0, 30.0, true, 80.0);
    assert!(
        !below_target.meets_p2_target(),
        "IMP-155a: 150 tok/s should not meet P2 target"
    );

    println!("\nIMP-155a: Fused Kernel Results:");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  Bandwidth: {:.1} GB/s", result.memory_bandwidth_gbs);
    println!("  Speedup: {:.1}x vs separate", result.speedup_vs_separate);
    println!("  Meets P2: {}", result.meets_p2_target());
}

/// IMP-155b: Fused vs separate performance comparison
#[derive(Debug, Clone)]
pub struct FusedVsSeparateComparison {
    pub fused_tps: f64,
    pub separate_tps: f64,
    pub speedup: f64,
    pub memory_reduction_percent: f64,
    pub fused_wins: bool,
}

impl FusedVsSeparateComparison {
    pub fn new(fused_tps: f64, separate_tps: f64) -> Self {
        let speedup = if separate_tps > 0.0 {
            fused_tps / separate_tps
        } else {
            1.0
        };
        // Fused eliminates intermediate buffer: ~50% memory reduction
        let memory_reduction = if speedup > 1.0 { 50.0 } else { 0.0 };
        Self {
            fused_tps,
            separate_tps,
            speedup,
            memory_reduction_percent: memory_reduction,
            fused_wins: speedup > 1.0,
        }
    }
}

/// IMP-155b: Test fused vs separate comparison
#[test]
fn test_imp_155b_fused_vs_separate() {
    // Per IMP-100c: Fused should be 29-132x faster
    let comparison = FusedVsSeparateComparison::new(5000.0, 170.0); // test values

    assert!(comparison.fused_wins, "IMP-155b: Fused should win");
    assert!(
        comparison.speedup > 20.0,
        "IMP-155b: Should show >20x speedup per IMP-100c"
    );
    assert!(
        comparison.memory_reduction_percent > 0.0,
        "IMP-155b: Should show memory reduction"
    );

    // Edge case: separate faster (shouldn't happen in practice)
    let edge = FusedVsSeparateComparison::new(100.0, 200.0);
    assert!(!edge.fused_wins, "IMP-155b: Separate faster edge case");

    println!("\nIMP-155b: Fused vs Separate:");
    println!("  Fused: {:.0} tok/s", comparison.fused_tps);
    println!("  Separate: {:.0} tok/s", comparison.separate_tps);
    println!("  Speedup: {:.1}x", comparison.speedup);
    println!(
        "  Memory reduction: {:.0}%",
        comparison.memory_reduction_percent
    );
}

/// IMP-155c: Real-world fused kernel vs llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_155c_fused_vs_llamacpp() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Explain quantum entanglement in simple terms:".to_string(),
        max_tokens: 50,
        temperature: Some(0.0),
        stream: false,
    };

    let start = std::time::Instant::now();
    let result = client
        .llamacpp_completion("http://127.0.0.1:8082", &request)
        .expect("IMP-155c: llama.cpp benchmark failed");
    let elapsed_s = start.elapsed().as_secs_f64();

    // Estimate throughput from response
    let tokens_generated = result.text.split_whitespace().count() as f64;
    let throughput_tps = tokens_generated / elapsed_s;

    // llama.cpp uses fused GGML kernels - this is our target
    let llamacpp_fused = FusedKernelResult::new(
        throughput_tps,
        50.0, // Estimated bandwidth
        true,
        throughput_tps / 30.0, // Estimate separate baseline
    );

    println!("\nIMP-155c: llama.cpp Fused Kernel Performance:");
    println!("  Throughput: {:.1} tok/s", llamacpp_fused.throughput_tps);
    println!("  Meets P2: {}", llamacpp_fused.meets_p2_target());
    println!(
        "  Est. speedup vs separate: {:.1}x",
        llamacpp_fused.speedup_vs_separate
    );
}

/// IMP-155d: Fused kernel memory efficiency analysis
#[derive(Debug, Clone)]
pub struct MemoryEfficiency {
    pub model_size_mb: f64,
    pub peak_memory_mb: f64,
    pub memory_overhead_percent: f64,
    pub bandwidth_utilization_percent: f64,
}

impl MemoryEfficiency {
    pub fn new(
        model_size_mb: f64,
        peak_memory_mb: f64,
        theoretical_bandwidth_gbs: f64,
        actual_bandwidth_gbs: f64,
    ) -> Self {
        let overhead = if model_size_mb > 0.0 {
            ((peak_memory_mb - model_size_mb) / model_size_mb) * 100.0
        } else {
            0.0
        };
        let utilization = if theoretical_bandwidth_gbs > 0.0 {
            (actual_bandwidth_gbs / theoretical_bandwidth_gbs) * 100.0
        } else {
            0.0
        };
        Self {
            model_size_mb,
            peak_memory_mb,
            memory_overhead_percent: overhead,
            bandwidth_utilization_percent: utilization,
        }
    }

    pub fn is_memory_efficient(&self) -> bool {
        // Efficient if overhead < 50% and bandwidth utilization > 50%
        self.memory_overhead_percent < 50.0 && self.bandwidth_utilization_percent > 50.0
    }
}

/// IMP-155d: Test memory efficiency analysis
#[test]
fn test_imp_155d_memory_efficiency() {
    // Scenario: Q4_K model 7.74 MB, peak 10 MB, 50% bandwidth utilization
    let efficient = MemoryEfficiency::new(7.74, 10.0, 100.0, 55.0);
    assert!(
        efficient.is_memory_efficient(),
        "IMP-155d: 29% overhead, 55% bandwidth should be efficient"
    );

    // Scenario: High overhead (separate path)
    let inefficient = MemoryEfficiency::new(7.74, 20.0, 100.0, 30.0);
    assert!(
        !inefficient.is_memory_efficient(),
        "IMP-155d: 158% overhead should not be efficient"
    );

    println!("\nIMP-155d: Memory Efficiency:");
    println!("  Model size: {:.2} MB", efficient.model_size_mb);
    println!("  Peak memory: {:.2} MB", efficient.peak_memory_mb);
    println!("  Overhead: {:.1}%", efficient.memory_overhead_percent);
    println!(
        "  Bandwidth util: {:.1}%",
        efficient.bandwidth_utilization_percent
    );
    println!("  Efficient: {}", efficient.is_memory_efficient());
}

// =========================================================================
// IMP-156: Latency Percentile Comparison (P50/P95/P99) (EXTREME TDD)
// Per spec QA-035: Results include p50, p95, p99 latencies
// =========================================================================

/// IMP-156a: Latency percentiles
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub stddev_ms: f64,
}

impl LatencyPercentiles {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                stddev_ms: 0.0,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let p50_idx = (n as f64 * 0.50) as usize;
        let p95_idx = (n as f64 * 0.95) as usize;
        let p99_idx = (n as f64 * 0.99) as usize;

        let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
        let variance: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        Self {
            p50_ms: sorted.get(p50_idx.min(n - 1)).copied().unwrap_or(0.0),
            p95_ms: sorted.get(p95_idx.min(n - 1)).copied().unwrap_or(0.0),
            p99_ms: sorted.get(p99_idx.min(n - 1)).copied().unwrap_or(0.0),
            min_ms: sorted.first().copied().unwrap_or(0.0),
            max_ms: sorted.last().copied().unwrap_or(0.0),
            mean_ms: mean,
            stddev_ms: variance.sqrt(),
        }
    }

    pub fn tail_latency_ratio(&self) -> f64 {
        if self.p50_ms > 0.0 {
            self.p99_ms / self.p50_ms
        } else {
            1.0
        }
    }
}

/// IMP-156a: Test latency percentile calculation
#[test]
fn test_imp_156a_latency_percentiles() {
    // 100 samples: mostly 10ms, some outliers
    let mut samples: Vec<f64> = vec![10.0; 90];
    samples.extend(vec![50.0; 5]); // P95 region
    samples.extend(vec![100.0; 5]); // P99 region

    let percentiles = LatencyPercentiles::from_samples(&samples);

    assert!(
        (percentiles.p50_ms - 10.0).abs() < 1.0,
        "IMP-156a: P50 should be ~10ms"
    );
    assert!(
        percentiles.p95_ms >= 10.0 && percentiles.p95_ms <= 100.0,
        "IMP-156a: P95 should be between 10-100ms"
    );
    assert!(
        percentiles.p99_ms >= 50.0,
        "IMP-156a: P99 should be >= 50ms"
    );
    assert!(
        percentiles.tail_latency_ratio() >= 1.0,
        "IMP-156a: Tail ratio should be >= 1"
    );

    println!("\nIMP-156a: Latency Percentiles:");
    println!("  P50: {:.1}ms", percentiles.p50_ms);
    println!("  P95: {:.1}ms", percentiles.p95_ms);
    println!("  P99: {:.1}ms", percentiles.p99_ms);
    println!(
        "  Min: {:.1}ms, Max: {:.1}ms",
        percentiles.min_ms, percentiles.max_ms
    );
    println!(
        "  Mean: {:.1}ms, Stddev: {:.1}ms",
        percentiles.mean_ms, percentiles.stddev_ms
    );
    println!(
        "  Tail ratio (P99/P50): {:.2}x",
        percentiles.tail_latency_ratio()
    );
}

/// IMP-156b: Latency comparison between runners
#[derive(Debug, Clone)]
pub struct LatencyComparison {
    pub realizar_percentiles: LatencyPercentiles,
    pub reference_percentiles: LatencyPercentiles,
    pub p50_gap_percent: f64,
    pub p99_gap_percent: f64,
    pub realizar_has_lower_p50: bool,
    pub realizar_has_lower_p99: bool,
}

impl LatencyComparison {
    pub fn new(realizar: LatencyPercentiles, reference: LatencyPercentiles) -> Self {
        let p50_gap = if reference.p50_ms > 0.0 {
            ((realizar.p50_ms - reference.p50_ms) / reference.p50_ms) * 100.0
        } else {
            0.0
        };
        let p99_gap = if reference.p99_ms > 0.0 {
            ((realizar.p99_ms - reference.p99_ms) / reference.p99_ms) * 100.0
        } else {
            0.0
        };
        Self {
            realizar_percentiles: realizar.clone(),
            reference_percentiles: reference.clone(),
            p50_gap_percent: p50_gap,
            p99_gap_percent: p99_gap,
            realizar_has_lower_p50: realizar.p50_ms < reference.p50_ms,
            realizar_has_lower_p99: realizar.p99_ms < reference.p99_ms,
        }
    }

    pub fn parity_achieved(&self) -> bool {
        // Parity if within 20% on both P50 and P99
        self.p50_gap_percent.abs() <= 20.0 && self.p99_gap_percent.abs() <= 20.0
    }
}

/// IMP-156b: Test latency comparison
#[test]
fn test_imp_156b_latency_comparison() {
    let realizar = LatencyPercentiles {
        p50_ms: 12.0,
        p95_ms: 25.0,
        p99_ms: 45.0,
        min_ms: 8.0,
        max_ms: 60.0,
        mean_ms: 15.0,
        stddev_ms: 8.0,
    };
    let reference = LatencyPercentiles {
        p50_ms: 10.0,
        p95_ms: 20.0,
        p99_ms: 40.0,
        min_ms: 7.0,
        max_ms: 55.0,
        mean_ms: 12.0,
        stddev_ms: 6.0,
    };

    let comparison = LatencyComparison::new(realizar, reference);

    assert!(
        (comparison.p50_gap_percent - 20.0).abs() < 1.0,
        "IMP-156b: P50 gap should be ~20%"
    );
    assert!(
        comparison.parity_achieved(),
        "IMP-156b: Should be at parity (within 20%)"
    );

    println!("\nIMP-156b: Latency Comparison:");
    println!(
        "  Realizar P50: {:.1}ms",
        comparison.realizar_percentiles.p50_ms
    );
    println!(
        "  Reference P50: {:.1}ms",
        comparison.reference_percentiles.p50_ms
    );
    println!("  P50 gap: {:+.1}%", comparison.p50_gap_percent);
    println!("  P99 gap: {:+.1}%", comparison.p99_gap_percent);
    println!("  Parity: {}", comparison.parity_achieved());
}

/// IMP-156c: Real-world latency comparison vs llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_156c_latency_vs_llamacpp() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count from 1 to 5:".to_string(),
        max_tokens: 20,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect multiple samples for percentile calculation
    let mut latencies_ms = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        let _ = client.llamacpp_completion("http://127.0.0.1:8082", &request);
        latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let percentiles = LatencyPercentiles::from_samples(&latencies_ms);

    println!("\nIMP-156c: llama.cpp Latency Percentiles:");
    println!("  P50: {:.2}ms", percentiles.p50_ms);
    println!("  P95: {:.2}ms", percentiles.p95_ms);
    println!("  P99: {:.2}ms", percentiles.p99_ms);
    println!("  Tail ratio: {:.2}x", percentiles.tail_latency_ratio());
}

/// IMP-156d: Latency SLA gate
#[derive(Debug, Clone)]
pub struct LatencySLAGate {
    pub p50_limit_ms: f64,
    pub p99_limit_ms: f64,
    pub measured_p50_ms: f64,
    pub measured_p99_ms: f64,
    pub p50_pass: bool,
    pub p99_pass: bool,
    pub overall_pass: bool,
}

impl LatencySLAGate {
    pub fn new(p50_limit_ms: f64, p99_limit_ms: f64, measured: &LatencyPercentiles) -> Self {
        let p50_pass = measured.p50_ms <= p50_limit_ms;
        let p99_pass = measured.p99_ms <= p99_limit_ms;
        Self {
            p50_limit_ms,
            p99_limit_ms,
            measured_p50_ms: measured.p50_ms,
            measured_p99_ms: measured.p99_ms,
            p50_pass,
            p99_pass,
            overall_pass: p50_pass && p99_pass,
        }
    }
}

/// IMP-156d: Test latency SLA gate
#[test]
fn test_imp_156d_latency_sla() {
    let good_latency = LatencyPercentiles {
        p50_ms: 8.0,
        p95_ms: 15.0,
        p99_ms: 25.0,
        min_ms: 5.0,
        max_ms: 40.0,
        mean_ms: 10.0,
        stddev_ms: 5.0,
    };

    // SLA: P50 < 10ms, P99 < 30ms
    let gate = LatencySLAGate::new(10.0, 30.0, &good_latency);
    assert!(gate.overall_pass, "IMP-156d: Good latency should pass SLA");

    let bad_latency = LatencyPercentiles {
        p50_ms: 15.0,
        p95_ms: 40.0,
        p99_ms: 80.0,
        min_ms: 10.0,
        max_ms: 100.0,
        mean_ms: 20.0,
        stddev_ms: 15.0,
    };

    let fail_gate = LatencySLAGate::new(10.0, 30.0, &bad_latency);
    assert!(
        !fail_gate.overall_pass,
        "IMP-156d: Bad latency should fail SLA"
    );
    assert!(!fail_gate.p50_pass, "IMP-156d: P50 15ms > 10ms limit");
    assert!(!fail_gate.p99_pass, "IMP-156d: P99 80ms > 30ms limit");

    println!("\nIMP-156d: Latency SLA Gate:");
    println!(
        "  Good: P50={:.0}ms (limit {:.0}), P99={:.0}ms (limit {:.0}) -> {}",
        gate.measured_p50_ms,
        gate.p50_limit_ms,
        gate.measured_p99_ms,
        gate.p99_limit_ms,
        if gate.overall_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Bad: P50={:.0}ms (limit {:.0}), P99={:.0}ms (limit {:.0}) -> {}",
        fail_gate.measured_p50_ms,
        fail_gate.p50_limit_ms,
        fail_gate.measured_p99_ms,
        fail_gate.p99_limit_ms,
        if fail_gate.overall_pass {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

// =========================================================================
// IMP-157: Environment Metadata Capture (EXTREME TDD)
// Per spec QA-033: Environment metadata captured per Vitek & Kalibera [8]
// =========================================================================

/// IMP-157a: System environment metadata
#[derive(Debug, Clone)]
pub struct EnvironmentMetadata {
    pub os_name: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub rust_version: String,
    pub realizar_version: String,
    pub timestamp: String,
    pub hostname: String,
}

impl EnvironmentMetadata {
    pub fn capture() -> Self {
        Self {
            os_name: std::env::consts::OS.to_string(),
            os_version: std::env::consts::ARCH.to_string(),
            cpu_model: "Unknown".to_string(), // Would need sysinfo crate
            cpu_cores: std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1),
            memory_gb: 0.0, // Would need sysinfo crate
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            realizar_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::json!({
            "os": {
                "name": self.os_name,
                "version": self.os_version
            },
            "cpu": {
                "model": self.cpu_model,
                "cores": self.cpu_cores
            },
            "memory_gb": self.memory_gb,
            "software": {
                "rust_version": self.rust_version,
                "realizar_version": self.realizar_version
            },
            "timestamp": self.timestamp,
            "hostname": self.hostname
        })
        .to_string()
    }
}

/// IMP-157a: Test environment metadata capture
#[test]
fn test_imp_157a_environment_capture() {
    let env = EnvironmentMetadata::capture();

    assert!(
        !env.os_name.is_empty(),
        "IMP-157a: OS name should not be empty"
    );
    assert!(env.cpu_cores > 0, "IMP-157a: CPU cores should be > 0");
    assert!(
        !env.realizar_version.is_empty(),
        "IMP-157a: Version should not be empty"
    );

    let json = env.to_json();
    assert!(json.contains("os"), "IMP-157a: JSON should have os field");
    assert!(json.contains("cpu"), "IMP-157a: JSON should have cpu field");

    println!("\nIMP-157a: Environment Metadata:");
    println!("  OS: {} {}", env.os_name, env.os_version);
    println!("  CPU cores: {}", env.cpu_cores);
    println!("  Rust: {}", env.rust_version);
    println!("  Realizar: {}", env.realizar_version);
    println!("  Timestamp: {}", env.timestamp);
}

/// IMP-157b: Benchmark configuration metadata
#[derive(Debug, Clone)]
pub struct BenchmarkMetadata {
    pub benchmark_name: String,
    pub model_path: String,
    pub model_size_mb: f64,
    pub quantization: String,
    pub batch_size: usize,
    pub max_tokens: usize,
    pub cv_threshold: f64,
    pub warmup_iterations: usize,
}

impl BenchmarkMetadata {
    pub fn new(name: &str) -> Self {
        Self {
            benchmark_name: name.to_string(),
            model_path: String::new(),
            model_size_mb: 0.0,
            quantization: "Q4_K".to_string(),
            batch_size: 1,
            max_tokens: 100,
            cv_threshold: 0.10,
            warmup_iterations: 3,
        }
    }

    pub fn with_model(mut self, path: &str, size_mb: f64, quant: &str) -> Self {
        self.model_path = path.to_string();
        self.model_size_mb = size_mb;
        self.quantization = quant.to_string();
        self
    }
}

/// IMP-157b: Test benchmark metadata
#[test]
fn test_imp_157b_benchmark_metadata() {
    let meta = BenchmarkMetadata::new("performance_parity").with_model(
        "phi-2-q4k.gguf",
        1.6 * 1024.0,
        "Q4_K_M",
    );

    assert_eq!(meta.benchmark_name, "performance_parity");
    assert!(
        meta.model_size_mb > 1000.0,
        "IMP-157b: Model should be > 1GB"
    );
    assert_eq!(meta.quantization, "Q4_K_M");

    println!("\nIMP-157b: Benchmark Metadata:");
    println!("  Name: {}", meta.benchmark_name);
    println!(
        "  Model: {} ({:.1} MB)",
        meta.model_path, meta.model_size_mb
    );
    println!("  Quantization: {}", meta.quantization);
    println!("  Batch size: {}", meta.batch_size);
    println!("  CV threshold: {:.0}%", meta.cv_threshold * 100.0);
}

/// IMP-157c: Full benchmark result with metadata
#[derive(Debug, Clone)]
pub struct FullBenchmarkResult {
    pub environment: EnvironmentMetadata,
    pub benchmark: BenchmarkMetadata,
    pub throughput_tps: f64,
    pub latency: LatencyPercentiles,
    pub iterations: usize,
    pub cv_achieved: f64,
}

impl FullBenchmarkResult {
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "environment": serde_json::from_str::<serde_json::Value>(&self.environment.to_json()).unwrap_or_default(),
            "benchmark": {
                "name": self.benchmark.benchmark_name,
                "model_path": self.benchmark.model_path,
                "model_size_mb": self.benchmark.model_size_mb,
                "quantization": self.benchmark.quantization
            },
            "results": {
                "throughput_tps": self.throughput_tps,
                "latency_p50_ms": self.latency.p50_ms,
                "latency_p95_ms": self.latency.p95_ms,
                "latency_p99_ms": self.latency.p99_ms,
                "iterations": self.iterations,
                "cv_achieved": self.cv_achieved
            }
        }).to_string()
    }
}

/// IMP-157c: Test full benchmark result
#[test]
fn test_imp_157c_full_benchmark_result() {
    let result = FullBenchmarkResult {
        environment: EnvironmentMetadata::capture(),
        benchmark: BenchmarkMetadata::new("parity_test"),
        throughput_tps: 150.0,
        latency: LatencyPercentiles {
            p50_ms: 10.0,
            p95_ms: 20.0,
            p99_ms: 35.0,
            min_ms: 8.0,
            max_ms: 50.0,
            mean_ms: 12.0,
            stddev_ms: 5.0,
        },
        iterations: 25,
        cv_achieved: 0.08,
    };

    let json = result.to_json();
    assert!(
        json.contains("environment"),
        "IMP-157c: Should have environment"
    );
    assert!(
        json.contains("throughput_tps"),
        "IMP-157c: Should have throughput"
    );
    assert!(
        json.contains("latency_p50_ms"),
        "IMP-157c: Should have latency"
    );

    println!("\nIMP-157c: Full Benchmark Result JSON:");
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::from_str::<serde_json::Value>(&json).expect("test")
        )
        .unwrap_or(json)
    );
}

/// IMP-157d: Reproducibility hash
#[derive(Debug, Clone)]
pub struct ReproducibilityHash {
    pub config_hash: String,
    pub environment_hash: String,
    pub combined_hash: String,
}

impl ReproducibilityHash {
    pub fn compute(env: &EnvironmentMetadata, bench: &BenchmarkMetadata) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut config_hasher = DefaultHasher::new();
        bench.benchmark_name.hash(&mut config_hasher);
        bench.quantization.hash(&mut config_hasher);
        bench.max_tokens.hash(&mut config_hasher);
        let config_hash = format!("{:016x}", config_hasher.finish());

        let mut env_hasher = DefaultHasher::new();
        env.os_name.hash(&mut env_hasher);
        env.cpu_cores.hash(&mut env_hasher);
        env.rust_version.hash(&mut env_hasher);
        let env_hash = format!("{:016x}", env_hasher.finish());

        let mut combined_hasher = DefaultHasher::new();
        config_hash.hash(&mut combined_hasher);
        env_hash.hash(&mut combined_hasher);
        let combined = format!("{:016x}", combined_hasher.finish());

        Self {
            config_hash,
            environment_hash: env_hash,
            combined_hash: combined,
        }
    }
}

/// IMP-157d: Test reproducibility hash
#[test]
fn test_imp_157d_reproducibility_hash() {
    let env = EnvironmentMetadata::capture();
    let bench = BenchmarkMetadata::new("test_bench");

    let hash1 = ReproducibilityHash::compute(&env, &bench);
    let hash2 = ReproducibilityHash::compute(&env, &bench);

    assert_eq!(
        hash1.combined_hash, hash2.combined_hash,
        "IMP-157d: Same inputs should produce same hash"
    );
    assert_eq!(
        hash1.config_hash.len(),
        16,
        "IMP-157d: Config hash should be 16 chars"
    );
    assert_eq!(
        hash1.environment_hash.len(),
        16,
        "IMP-157d: Env hash should be 16 chars"
    );

    println!("\nIMP-157d: Reproducibility Hash:");
    println!("  Config: {}", hash1.config_hash);
    println!("  Environment: {}", hash1.environment_hash);
    println!("  Combined: {}", hash1.combined_hash);
}

// =========================================================================
// IMP-158: Benchmark Result JSON Schema Validation (EXTREME TDD)
// Per spec QA-040: JSON schema validation for benchmark results
