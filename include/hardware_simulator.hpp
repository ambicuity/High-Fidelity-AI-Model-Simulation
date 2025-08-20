#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>

namespace HighFidelityAI {

// Forward declarations
struct MemoryOperation;
struct ComputeOperation;

/**
 * Hardware configuration parameters
 */
struct HardwareConfig {
    // CPU parameters
    size_t num_cores = 8;
    double cpu_frequency_ghz = 3.2;
    size_t cache_l1_size_kb = 32;
    size_t cache_l2_size_kb = 256;
    size_t cache_l3_size_mb = 8;
    
    // Memory parameters  
    size_t memory_capacity_gb = 32;
    double memory_bandwidth_gbps = 100.0;
    double memory_latency_ns = 100.0;
    
    // Accelerator parameters
    size_t num_accelerators = 4;
    double accelerator_compute_tflops = 50.0;
    double accelerator_memory_bandwidth_gbps = 900.0;
    
    // Interconnect parameters
    double interconnect_bandwidth_gbps = 25.0;
    double interconnect_latency_ns = 500.0;
};

/**
 * Performance metrics collected during simulation
 */
struct PerformanceMetrics {
    double total_execution_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double memory_time_ms = 0.0;
    double communication_time_ms = 0.0;
    double energy_consumption_j = 0.0;
    size_t memory_accesses = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    double utilization_cpu = 0.0;
    double utilization_accelerator = 0.0;
    
    void reset();
    void print_summary() const;
};

/**
 * CPU core simulator with cache hierarchy
 */
class CPUCore {
public:
    explicit CPUCore(size_t core_id, const HardwareConfig& config);
    
    double execute_compute_operation(const ComputeOperation& op);
    double execute_memory_operation(const MemoryOperation& op);
    
    void reset_metrics();
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    
private:
    size_t core_id_;
    HardwareConfig config_;
    PerformanceMetrics metrics_;
    
    // Cache simulation
    std::unordered_map<size_t, bool> l1_cache_;
    std::unordered_map<size_t, bool> l2_cache_;
    
    bool check_cache_hit(size_t address, int level);
    void update_cache(size_t address, int level);
};

/**
 * Accelerator (GPU/TPU) simulator
 */
class Accelerator {
public:
    explicit Accelerator(size_t accelerator_id, const HardwareConfig& config);
    
    double execute_parallel_operation(const ComputeOperation& op);
    double execute_memory_transfer(const MemoryOperation& op);
    
    void reset_metrics();
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    
private:
    size_t accelerator_id_;
    HardwareConfig config_;
    PerformanceMetrics metrics_;
    
    double calculate_parallel_efficiency(size_t num_operations) const;
};

/**
 * Memory subsystem simulator
 */
class MemorySubsystem {
public:
    explicit MemorySubsystem(const HardwareConfig& config);
    
    double simulate_memory_access(size_t address, size_t size, bool is_write = false);
    double simulate_memory_transfer(size_t size, bool to_accelerator = false);
    
    void reset_metrics();
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    
private:
    HardwareConfig config_;
    PerformanceMetrics metrics_;
    
    // Memory bank conflict modeling
    std::vector<double> bank_busy_until_;
    
    size_t get_memory_bank(size_t address) const;
    double calculate_bank_conflict_delay(size_t bank) const;
};

/**
 * Main hardware simulator orchestrating all components
 */
class HardwareSimulator {
public:
    explicit HardwareSimulator(const HardwareConfig& config = HardwareConfig{});
    
    void set_config(const HardwareConfig& config);
    const HardwareConfig& get_config() const { return config_; }
    
    // Execution simulation
    double simulate_compute_operation(const ComputeOperation& op, size_t preferred_core = 0);
    double simulate_memory_operation(const MemoryOperation& op);
    double simulate_parallel_operation(const ComputeOperation& op, size_t preferred_accelerator = 0);
    
    // Metrics and analysis
    void reset_all_metrics();
    PerformanceMetrics get_aggregate_metrics() const;
    void print_detailed_metrics() const;
    
    // Hardware optimization recommendations
    std::vector<std::string> analyze_bottlenecks() const;
    std::vector<std::string> suggest_optimizations() const;
    
private:
    HardwareConfig config_;
    std::vector<std::unique_ptr<CPUCore>> cpu_cores_;
    std::vector<std::unique_ptr<Accelerator>> accelerators_;
    std::unique_ptr<MemorySubsystem> memory_subsystem_;
    
    void initialize_hardware_components();
    size_t select_optimal_core() const;
    size_t select_optimal_accelerator() const;
};

} // namespace HighFidelityAI