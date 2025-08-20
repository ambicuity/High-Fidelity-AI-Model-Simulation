#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace HighFidelityAI {

// Forward declarations
struct PerformanceMetrics;
struct HardwareConfig;
struct ModelConfig;
class HardwareSimulator;
class AIModelSimulator;
class PerformanceProfiler;
struct OptimizationResult;
struct MultiObjectiveResult;

/**
 * Analysis categories for performance bottlenecks
 */
enum class BottleneckType {
    COMPUTE_BOUND,
    MEMORY_BOUND,
    COMMUNICATION_BOUND,
    CACHE_BOUND,
    BANDWIDTH_BOUND,
    PARALLELIZATION_BOUND
};

/**
 * Performance bottleneck analysis
 */
struct BottleneckAnalysis {
    BottleneckType type;
    std::string description;
    double severity_score;      // 0.0 to 1.0
    std::string location;       // Hardware component or model layer
    std::vector<std::string> recommendations;
    double potential_improvement; // Estimated speedup if addressed
    
    BottleneckAnalysis(BottleneckType t, const std::string& desc, double severity)
        : type(t), description(desc), severity_score(severity) {}
};

/**
 * Hardware utilization analysis
 */
struct UtilizationAnalysis {
    std::unordered_map<std::string, double> component_utilization;
    std::vector<std::string> underutilized_components;
    std::vector<std::string> overutilized_components;
    double overall_efficiency_score = 0.0;
    
    void print_summary() const;
};

/**
 * Model efficiency analysis
 */
struct ModelEfficiencyAnalysis {
    std::unordered_map<std::string, double> layer_efficiency;
    std::vector<std::string> inefficient_layers;
    double arithmetic_intensity = 0.0;
    double memory_efficiency = 0.0;
    size_t total_parameters = 0;
    size_t effective_parameters = 0;  // Parameters actually used
    
    void print_summary() const;
};

/**
 * Energy consumption analysis
 */
struct EnergyAnalysis {
    double total_energy_j = 0.0;
    std::unordered_map<std::string, double> component_energy_breakdown;
    double energy_efficiency_gflops_per_watt = 0.0;
    std::vector<std::string> energy_hotspots;
    std::vector<std::string> energy_optimization_suggestions;
    
    void print_summary() const;
};

/**
 * Scalability analysis results
 */
struct ScalabilityAnalysis {
    struct ScalingPoint {
        size_t scale_factor;
        double performance_ratio;
        double efficiency_ratio;
        double resource_utilization;
    };
    
    std::vector<ScalingPoint> scaling_curve;
    double parallel_efficiency_factor = 0.0;
    size_t optimal_scale_factor = 1;
    std::string scaling_bottleneck;
    
    void print_summary() const;
};

/**
 * Comprehensive performance analyzer
 */
class PerformanceAnalyzer {
public:
    PerformanceAnalyzer() = default;
    
    // Core analysis functions
    std::vector<BottleneckAnalysis> identify_bottlenecks(
        const PerformanceMetrics& metrics,
        const HardwareConfig& hw_config,
        const ModelConfig& model_config) const;
    
    UtilizationAnalysis analyze_hardware_utilization(
        const HardwareSimulator& hw_sim) const;
    
    ModelEfficiencyAnalysis analyze_model_efficiency(
        const AIModelSimulator& model_sim,
        const PerformanceMetrics& metrics) const;
    
    EnergyAnalysis analyze_energy_consumption(
        const PerformanceMetrics& metrics,
        const HardwareConfig& hw_config) const;
    
    ScalabilityAnalysis analyze_scalability(
        HardwareSimulator& hw_sim,
        AIModelSimulator& model_sim,
        const std::vector<size_t>& scale_factors = {1, 2, 4, 8, 16}) const;
    
    // Roofline model analysis
    void generate_roofline_analysis(
        const AIModelSimulator& model_sim,
        const HardwareConfig& hw_config,
        const std::string& output_file = "roofline.csv") const;
    
    // Comprehensive analysis report
    void generate_comprehensive_report(
        const HardwareSimulator& hw_sim,
        const AIModelSimulator& model_sim,
        const PerformanceProfiler& profiler,
        const std::string& output_dir = "./analysis") const;
    
private:
    // Helper functions for bottleneck detection
    bool is_compute_bound(const PerformanceMetrics& metrics) const;
    bool is_memory_bound(const PerformanceMetrics& metrics) const;
    bool is_cache_bound(const PerformanceMetrics& metrics) const;
    bool is_bandwidth_bound(const PerformanceMetrics& metrics, const HardwareConfig& config) const;
    
    double calculate_severity(const PerformanceMetrics& metrics, BottleneckType type) const;
    std::vector<std::string> generate_recommendations(BottleneckType type, const std::string& location) const;
    
    // Analysis utilities
    double calculate_arithmetic_intensity(const AIModelSimulator& model_sim) const;
    double calculate_memory_efficiency(const PerformanceMetrics& metrics) const;
    
    void create_analysis_directory(const std::string& dir) const;
    void write_csv_report(const std::string& filename, const std::vector<std::vector<std::string>>& data) const;
};

/**
 * Optimization recommendation engine
 */
class RecommendationEngine {
public:
    RecommendationEngine() = default;
    
    // Generate hardware recommendations
    std::vector<std::string> generate_hardware_recommendations(
        const std::vector<BottleneckAnalysis>& bottlenecks,
        const UtilizationAnalysis& utilization,
        const HardwareConfig& current_config) const;
    
    // Generate model architecture recommendations
    std::vector<std::string> generate_model_recommendations(
        const ModelEfficiencyAnalysis& efficiency,
        const std::vector<BottleneckAnalysis>& bottlenecks,
        const ModelConfig& current_config) const;
    
    // Generate algorithmic optimization recommendations
    std::vector<std::string> generate_algorithmic_recommendations(
        const PerformanceMetrics& metrics,
        const ModelConfig& model_config) const;
    
    // Generate system-level recommendations
    std::vector<std::string> generate_system_recommendations(
        const EnergyAnalysis& energy,
        const ScalabilityAnalysis& scalability) const;
    
    // Prioritize recommendations based on impact
    std::vector<std::string> prioritize_recommendations(
        const std::vector<std::string>& recommendations,
        const std::vector<BottleneckAnalysis>& bottlenecks) const;
    
    // Generate comprehensive optimization strategy
    void generate_optimization_strategy(
        const std::vector<BottleneckAnalysis>& bottlenecks,
        const UtilizationAnalysis& utilization,
        const ModelEfficiencyAnalysis& efficiency,
        const EnergyAnalysis& energy,
        const ScalabilityAnalysis& scalability,
        const std::string& output_file = "optimization_strategy.md") const;
    
private:
    // Recommendation templates
    std::unordered_map<BottleneckType, std::vector<std::string>> bottleneck_recommendations_;
    std::unordered_map<std::string, std::vector<std::string>> component_recommendations_;
    
    void initialize_recommendation_templates();
    double estimate_recommendation_impact(const std::string& recommendation,
                                        const std::vector<BottleneckAnalysis>& bottlenecks) const;
};

/**
 * Comparative analysis between different configurations
 */
class ComparativeAnalyzer {
public:
    ComparativeAnalyzer() = default;
    
    // Compare optimization results
    void compare_optimization_results(
        const std::vector<OptimizationResult>& results,
        const std::string& output_file = "comparison.html") const;
    
    // Compare different hardware configurations
    void compare_hardware_configurations(
        const std::vector<HardwareConfig>& configs,
        AIModelSimulator& model_sim,
        const std::string& output_file = "hardware_comparison.csv") const;
    
    // Compare different model architectures
    void compare_model_architectures(
        const std::vector<ModelConfig>& configs,
        HardwareSimulator& hw_sim,
        const std::string& output_file = "model_comparison.csv") const;
    
    // Pareto frontier analysis
    void analyze_pareto_frontier(
        const MultiObjectiveResult& results,
        const std::string& output_file = "pareto_analysis.html") const;
    
    // Generate trade-off analysis
    std::vector<std::string> analyze_tradeoffs(
        const std::vector<OptimizationResult>& results,
        const std::vector<std::string>& metrics_to_compare = {"latency", "energy", "throughput"}) const;
    
private:
    void generate_comparison_html(const std::string& filename, const std::vector<std::vector<std::string>>& data) const;
    void generate_comparison_csv(const std::string& filename, const std::vector<std::vector<std::string>>& data) const;
    
    std::vector<std::string> extract_key_metrics(const OptimizationResult& result) const;
};

} // namespace HighFidelityAI