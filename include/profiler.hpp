#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>

namespace HighFidelityAI {

// Forward declarations
struct PerformanceMetrics;
class HardwareSimulator;
class AIModelSimulator;

/**
 * Profiling event types
 */
enum class EventType {
    COMPUTE_START,
    COMPUTE_END,
    MEMORY_START,
    MEMORY_END,
    LAYER_START,
    LAYER_END,
    MODEL_START,
    MODEL_END
};

/**
 * Profiling event
 */
struct ProfilingEvent {
    EventType type;
    std::string name;
    std::chrono::high_resolution_clock::time_point timestamp;
    size_t thread_id = 0;
    std::unordered_map<std::string, double> metadata;
    
    ProfilingEvent(EventType t, const std::string& n)
        : type(t), name(n), timestamp(std::chrono::high_resolution_clock::now()) {}
};

/**
 * Performance profiler for detailed analysis
 */
class PerformanceProfiler {
public:
    PerformanceProfiler() = default;
    
    // Event recording
    void start_event(EventType type, const std::string& name);
    void end_event(EventType type, const std::string& name);
    void record_instant_event(const std::string& name, 
                             const std::unordered_map<std::string, double>& metadata = {});
    
    // Analysis
    void analyze_events();
    void print_summary() const;
    void print_detailed_timeline() const;
    void export_chrome_trace(const std::string& filename) const;
    
    // Metrics
    double get_total_execution_time() const;
    double get_compute_time_ratio() const;
    double get_memory_time_ratio() const;
    
    void reset();
    
private:
    std::vector<ProfilingEvent> events_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> active_events_;
    
    // Analysis results
    double total_time_ms_ = 0.0;
    double compute_time_ms_ = 0.0;
    double memory_time_ms_ = 0.0;
    std::unordered_map<std::string, double> layer_times_;
    
    void calculate_durations();
    std::string format_duration(double duration_ms) const;
};

/**
 * System-wide performance monitoring
 */
class SystemMonitor {
public:
    SystemMonitor() = default;
    
    void start_monitoring();
    void stop_monitoring();
    
    // Hardware utilization tracking
    void record_cpu_utilization(size_t core_id, double utilization);
    void record_accelerator_utilization(size_t accelerator_id, double utilization);
    void record_memory_usage(size_t bytes_used, size_t bytes_total);
    void record_energy_consumption(double watts);
    
    // Analysis
    double get_average_cpu_utilization() const;
    double get_average_accelerator_utilization() const;
    double get_peak_memory_usage_ratio() const;
    double get_total_energy_consumption() const;
    
    void print_system_summary() const;
    void reset();
    
private:
    bool monitoring_active_ = false;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    
    // Utilization tracking
    std::vector<std::vector<double>> cpu_utilization_history_;
    std::vector<std::vector<double>> accelerator_utilization_history_;
    std::vector<double> memory_usage_history_;
    std::vector<double> energy_consumption_history_;
    
    double calculate_average(const std::vector<double>& values) const;
};

/**
 * Comprehensive simulation profiler combining performance and system monitoring
 */
class SimulationProfiler {
public:
    SimulationProfiler() = default;
    
    // Combined profiling interface
    void start_simulation_profiling();
    void end_simulation_profiling();
    
    void profile_model_execution(AIModelSimulator& model, HardwareSimulator& hw_sim);
    
    // Comprehensive analysis
    void generate_performance_report(const std::string& output_dir = "./reports") const;
    void generate_optimization_recommendations(const std::string& filename = "recommendations.txt") const;
    
    // Data access
    const PerformanceProfiler& get_performance_profiler() const { return perf_profiler_; }
    const SystemMonitor& get_system_monitor() const { return system_monitor_; }
    
    void reset_all();
    
private:
    PerformanceProfiler perf_profiler_;
    SystemMonitor system_monitor_;
    
    void create_output_directory(const std::string& dir) const;
    void write_performance_csv(const std::string& filename) const;
    void write_utilization_csv(const std::string& filename) const;
    void generate_html_report(const std::string& filename) const;
};

/**
 * RAII helper for automatic event timing
 */
class ScopedProfiler {
public:
    ScopedProfiler(PerformanceProfiler& profiler, EventType type, const std::string& name)
        : profiler_(profiler), type_(type), name_(name) {
        profiler_.start_event(type_, name_);
    }
    
    ~ScopedProfiler() {
        profiler_.end_event(type_, name_);
    }
    
private:
    PerformanceProfiler& profiler_;
    EventType type_;
    std::string name_;
};

// Convenience macro for scoped profiling
#define PROFILE_SCOPE(profiler, type, name) \
    ScopedProfiler _prof(profiler, type, name)

#define PROFILE_COMPUTE(profiler, name) \
    ScopedProfiler _prof(profiler, EventType::COMPUTE_START, name)

#define PROFILE_MEMORY(profiler, name) \
    ScopedProfiler _prof(profiler, EventType::MEMORY_START, name)

#define PROFILE_LAYER(profiler, name) \
    ScopedProfiler _prof(profiler, EventType::LAYER_START, name)

} // namespace HighFidelityAI