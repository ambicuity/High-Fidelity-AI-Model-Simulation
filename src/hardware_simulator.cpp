#include "hardware_simulator.hpp"
#include "ai_model.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

namespace HighFidelityAI {

void PerformanceMetrics::reset() {
    total_execution_time_ms = 0.0;
    compute_time_ms = 0.0;
    memory_time_ms = 0.0;
    communication_time_ms = 0.0;
    energy_consumption_j = 0.0;
    memory_accesses = 0;
    cache_hits = 0;
    cache_misses = 0;
    utilization_cpu = 0.0;
    utilization_accelerator = 0.0;
}

void PerformanceMetrics::print_summary() const {
    std::cout << "=== Performance Metrics Summary ===" << std::endl;
    std::cout << "Total Execution Time: " << total_execution_time_ms << " ms" << std::endl;
    std::cout << "Compute Time: " << compute_time_ms << " ms (" 
              << (compute_time_ms / total_execution_time_ms * 100) << "%)" << std::endl;
    std::cout << "Memory Time: " << memory_time_ms << " ms (" 
              << (memory_time_ms / total_execution_time_ms * 100) << "%)" << std::endl;
    std::cout << "Communication Time: " << communication_time_ms << " ms" << std::endl;
    std::cout << "Energy Consumption: " << energy_consumption_j << " J" << std::endl;
    std::cout << "Memory Accesses: " << memory_accesses << std::endl;
    std::cout << "Cache Hit Rate: " << (static_cast<double>(cache_hits) / (cache_hits + cache_misses) * 100) 
              << "%" << std::endl;
    std::cout << "CPU Utilization: " << utilization_cpu << "%" << std::endl;
    std::cout << "Accelerator Utilization: " << utilization_accelerator << "%" << std::endl;
}

// CPUCore Implementation
CPUCore::CPUCore(size_t core_id, const HardwareConfig& config)
    : core_id_(core_id), config_(config) {}

double CPUCore::execute_compute_operation(const ComputeOperation& op) {
    // Simulate compute execution time based on FLOPS and frequency
    double execution_time_ms = static_cast<double>(op.flops) / 
                              (config_.cpu_frequency_ghz * 1e9) * 1000.0;
    
    // Add some variability for realism
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(1.0, 0.05);
    execution_time_ms *= dist(gen);
    
    metrics_.compute_time_ms += execution_time_ms;
    metrics_.total_execution_time_ms += execution_time_ms;
    
    // Estimate energy consumption (simplified model)
    double power_watts = 50.0 + (config_.cpu_frequency_ghz - 2.0) * 20.0;
    metrics_.energy_consumption_j += power_watts * (execution_time_ms / 1000.0);
    
    return execution_time_ms;
}

double CPUCore::execute_memory_operation(const MemoryOperation& op) {
    // Simulate cache lookup
    bool cache_hit = check_cache_hit(op.address, 1); // L1 cache
    if (!cache_hit) {
        cache_hit = check_cache_hit(op.address, 2); // L2 cache
    }
    
    double access_time_ms;
    if (cache_hit) {
        access_time_ms = 0.001; // 1ns for cache hit converted to ms
        metrics_.cache_hits++;
    } else {
        access_time_ms = config_.memory_latency_ns / 1e6; // Convert ns to ms
        metrics_.cache_misses++;
        update_cache(op.address, 1);
        update_cache(op.address, 2);
    }
    
    // Account for memory bandwidth (convert bytes to GB, then to time)
    double size_gb = static_cast<double>(op.size) / (1024.0 * 1024.0 * 1024.0);
    double transfer_time_ms = (size_gb / config_.memory_bandwidth_gbps) * 1000.0;
    access_time_ms = std::max(access_time_ms, transfer_time_ms);
    
    metrics_.memory_time_ms += access_time_ms;
    metrics_.total_execution_time_ms += access_time_ms;
    metrics_.memory_accesses++;
    
    return access_time_ms;
}

void CPUCore::reset_metrics() {
    metrics_.reset();
    l1_cache_.clear();
    l2_cache_.clear();
}

bool CPUCore::check_cache_hit(size_t address, int level) {
    if (level == 1) {
        return l1_cache_.find(address) != l1_cache_.end();
    } else if (level == 2) {
        return l2_cache_.find(address) != l2_cache_.end();
    }
    return false;
}

void CPUCore::update_cache(size_t address, int level) {
    if (level == 1) {
        l1_cache_[address] = true;
        // Simple LRU eviction if cache is full
        if (l1_cache_.size() > config_.cache_l1_size_kb * 16) { // Assuming 64-byte cache lines
            l1_cache_.erase(l1_cache_.begin());
        }
    } else if (level == 2) {
        l2_cache_[address] = true;
        if (l2_cache_.size() > config_.cache_l2_size_kb * 16) {
            l2_cache_.erase(l2_cache_.begin());
        }
    }
}

// Accelerator Implementation
Accelerator::Accelerator(size_t accelerator_id, const HardwareConfig& config)
    : accelerator_id_(accelerator_id), config_(config) {}

double Accelerator::execute_parallel_operation(const ComputeOperation& op) {
    // Calculate theoretical execution time based on TFLOPS
    double theoretical_time_ms = static_cast<double>(op.flops) / 
                                (config_.accelerator_compute_tflops * 1e12) * 1000.0;
    
    // Apply parallel efficiency factor
    double parallel_efficiency = calculate_parallel_efficiency(op.flops);
    double actual_time_ms = theoretical_time_ms / parallel_efficiency;
    
    metrics_.compute_time_ms += actual_time_ms;
    metrics_.total_execution_time_ms += actual_time_ms;
    
    // Energy consumption (higher for accelerators)
    double power_watts = 200.0 + (config_.accelerator_compute_tflops - 10.0) * 50.0;
    metrics_.energy_consumption_j += power_watts * (actual_time_ms / 1000.0);
    
    return actual_time_ms;
}

double Accelerator::execute_memory_transfer(const MemoryOperation& op) {
    // High-bandwidth memory transfer (convert bytes to GB)
    double size_gb = static_cast<double>(op.size) / (1024.0 * 1024.0 * 1024.0);
    double transfer_time_ms = (size_gb / config_.accelerator_memory_bandwidth_gbps) * 1000.0;
    
    metrics_.memory_time_ms += transfer_time_ms;
    metrics_.total_execution_time_ms += transfer_time_ms;
    metrics_.memory_accesses++;
    
    return transfer_time_ms;
}

void Accelerator::reset_metrics() {
    metrics_.reset();
}

double Accelerator::calculate_parallel_efficiency(size_t num_operations) const {
    // Simplified parallel efficiency model based on Amdahl's law
    double sequential_fraction = 0.1;  // 10% sequential
    double parallel_fraction = 1.0 - sequential_fraction;
    
    // Assume perfect scaling up to a certain point
    size_t max_parallel_units = 1024;  // Simplified assumption
    size_t effective_units = std::min(num_operations / 1000, max_parallel_units);
    
    if (effective_units == 0) effective_units = 1;
    
    double speedup = 1.0 / (sequential_fraction + parallel_fraction / effective_units);
    return std::min(speedup / effective_units, 1.0);
}

// MemorySubsystem Implementation
MemorySubsystem::MemorySubsystem(const HardwareConfig& config)
    : config_(config), bank_busy_until_(16, 0.0) {}  // 16 memory banks

double MemorySubsystem::simulate_memory_access(size_t address, size_t size, bool is_write) {
    size_t bank = get_memory_bank(address);
    double bank_conflict_delay = calculate_bank_conflict_delay(bank);
    
    // Base access latency (convert ns to ms)
    double access_time_ms = config_.memory_latency_ns / 1e6;
    
    // Bandwidth limitation: size in bytes, bandwidth in GB/s
    // Convert: bytes -> GB -> time in seconds -> time in ms
    double size_gb = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
    double transfer_time_ms = (size_gb / config_.memory_bandwidth_gbps) * 1000.0;
    
    // Take the maximum of latency and bandwidth limit
    double total_time_ms = std::max(access_time_ms, transfer_time_ms) + bank_conflict_delay;
    
    // Update bank busy time
    bank_busy_until_[bank] = std::max(bank_busy_until_[bank], total_time_ms);
    
    metrics_.memory_time_ms += total_time_ms;
    metrics_.total_execution_time_ms += total_time_ms;
    metrics_.memory_accesses++;
    
    return total_time_ms;
}

double MemorySubsystem::simulate_memory_transfer(size_t size, bool to_accelerator) {
    double bandwidth = to_accelerator ? 
        config_.accelerator_memory_bandwidth_gbps : config_.memory_bandwidth_gbps;
    
    // Convert bytes to GB, then calculate time in seconds, then convert to ms
    double size_gb = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
    double transfer_time_ms = (size_gb / bandwidth) * 1000.0;
    
    metrics_.communication_time_ms += transfer_time_ms;
    metrics_.total_execution_time_ms += transfer_time_ms;
    
    return transfer_time_ms;
}

void MemorySubsystem::reset_metrics() {
    metrics_.reset();
    std::fill(bank_busy_until_.begin(), bank_busy_until_.end(), 0.0);
}

size_t MemorySubsystem::get_memory_bank(size_t address) const {
    // Simple bank mapping based on address
    return (address >> 6) % bank_busy_until_.size();  // 64-byte cache line size
}

double MemorySubsystem::calculate_bank_conflict_delay(size_t bank) const {
    // If bank is busy, add delay
    return std::max(0.0, bank_busy_until_[bank] - 0.0);  // Simplified
}

// HardwareSimulator Implementation
HardwareSimulator::HardwareSimulator(const HardwareConfig& config)
    : config_(config) {
    initialize_hardware_components();
}

void HardwareSimulator::set_config(const HardwareConfig& config) {
    config_ = config;
    initialize_hardware_components();
}

double HardwareSimulator::simulate_compute_operation(const ComputeOperation& op, size_t preferred_core) {
    if (preferred_core >= cpu_cores_.size()) {
        preferred_core = select_optimal_core();
    }
    
    return cpu_cores_[preferred_core]->execute_compute_operation(op);
}

double HardwareSimulator::simulate_memory_operation(const MemoryOperation& op) {
    return memory_subsystem_->simulate_memory_access(op.address, op.size, op.is_write);
}

double HardwareSimulator::simulate_parallel_operation(const ComputeOperation& op, size_t preferred_accelerator) {
    if (preferred_accelerator >= accelerators_.size()) {
        preferred_accelerator = select_optimal_accelerator();
    }
    
    return accelerators_[preferred_accelerator]->execute_parallel_operation(op);
}

void HardwareSimulator::reset_all_metrics() {
    for (auto& core : cpu_cores_) {
        core->reset_metrics();
    }
    for (auto& accelerator : accelerators_) {
        accelerator->reset_metrics();
    }
    memory_subsystem_->reset_metrics();
}

PerformanceMetrics HardwareSimulator::get_aggregate_metrics() const {
    PerformanceMetrics aggregate;
    
    // Aggregate CPU metrics
    for (const auto& core : cpu_cores_) {
        const auto& metrics = core->get_metrics();
        aggregate.compute_time_ms += metrics.compute_time_ms;
        aggregate.memory_time_ms += metrics.memory_time_ms;
        aggregate.energy_consumption_j += metrics.energy_consumption_j;
        aggregate.memory_accesses += metrics.memory_accesses;
        aggregate.cache_hits += metrics.cache_hits;
        aggregate.cache_misses += metrics.cache_misses;
    }
    
    // Aggregate accelerator metrics
    for (const auto& accelerator : accelerators_) {
        const auto& metrics = accelerator->get_metrics();
        aggregate.compute_time_ms += metrics.compute_time_ms;
        aggregate.memory_time_ms += metrics.memory_time_ms;
        aggregate.energy_consumption_j += metrics.energy_consumption_j;
        aggregate.memory_accesses += metrics.memory_accesses;
    }
    
    // Memory subsystem metrics
    const auto& mem_metrics = memory_subsystem_->get_metrics();
    aggregate.memory_time_ms += mem_metrics.memory_time_ms;
    aggregate.communication_time_ms += mem_metrics.communication_time_ms;
    aggregate.memory_accesses += mem_metrics.memory_accesses;
    
    // Calculate total execution time (max of all components)
    aggregate.total_execution_time_ms = std::max({
        aggregate.compute_time_ms, aggregate.memory_time_ms, aggregate.communication_time_ms
    });
    
    // Calculate utilization
    if (aggregate.total_execution_time_ms > 0) {
        aggregate.utilization_cpu = (aggregate.compute_time_ms / cpu_cores_.size()) / 
                                   aggregate.total_execution_time_ms * 100.0;
        aggregate.utilization_accelerator = (aggregate.compute_time_ms / accelerators_.size()) / 
                                           aggregate.total_execution_time_ms * 100.0;
    }
    
    return aggregate;
}

void HardwareSimulator::print_detailed_metrics() const {
    std::cout << "\n=== Detailed Hardware Metrics ===" << std::endl;
    
    std::cout << "\nCPU Cores:" << std::endl;
    for (size_t i = 0; i < cpu_cores_.size(); ++i) {
        std::cout << "Core " << i << ":" << std::endl;
        cpu_cores_[i]->get_metrics().print_summary();
    }
    
    std::cout << "\nAccelerators:" << std::endl;
    for (size_t i = 0; i < accelerators_.size(); ++i) {
        std::cout << "Accelerator " << i << ":" << std::endl;
        accelerators_[i]->get_metrics().print_summary();
    }
    
    std::cout << "\nMemory Subsystem:" << std::endl;
    memory_subsystem_->get_metrics().print_summary();
    
    std::cout << "\nAggregate Metrics:" << std::endl;
    get_aggregate_metrics().print_summary();
}

std::vector<std::string> HardwareSimulator::analyze_bottlenecks() const {
    std::vector<std::string> bottlenecks;
    auto metrics = get_aggregate_metrics();
    
    // Analyze compute vs memory bound
    if (metrics.memory_time_ms > metrics.compute_time_ms * 1.5) {
        bottlenecks.push_back("Memory-bound: Consider increasing memory bandwidth or adding more cache");
    }
    
    if (metrics.utilization_cpu < 50.0) {
        bottlenecks.push_back("Low CPU utilization: Consider increasing parallelism or reducing memory latency");
    }
    
    if (metrics.utilization_accelerator < 50.0 && accelerators_.size() > 0) {
        bottlenecks.push_back("Low accelerator utilization: Consider optimizing for parallel execution");
    }
    
    double cache_hit_rate = static_cast<double>(metrics.cache_hits) / 
                           (metrics.cache_hits + metrics.cache_misses);
    if (cache_hit_rate < 0.8) {
        bottlenecks.push_back("Low cache hit rate: Consider increasing cache size or improving data locality");
    }
    
    return bottlenecks;
}

std::vector<std::string> HardwareSimulator::suggest_optimizations() const {
    std::vector<std::string> suggestions;
    auto metrics = get_aggregate_metrics();
    
    suggestions.push_back("Increase memory bandwidth to reduce memory-bound operations");
    suggestions.push_back("Add more cache levels or increase cache sizes");
    suggestions.push_back("Optimize memory access patterns for better locality");
    suggestions.push_back("Consider using more accelerators for parallel workloads");
    suggestions.push_back("Implement prefetching mechanisms to hide memory latency");
    
    return suggestions;
}

void HardwareSimulator::initialize_hardware_components() {
    // Initialize CPU cores
    cpu_cores_.clear();
    for (size_t i = 0; i < config_.num_cores; ++i) {
        cpu_cores_.push_back(std::make_unique<CPUCore>(i, config_));
    }
    
    // Initialize accelerators
    accelerators_.clear();
    for (size_t i = 0; i < config_.num_accelerators; ++i) {
        accelerators_.push_back(std::make_unique<Accelerator>(i, config_));
    }
    
    // Initialize memory subsystem
    memory_subsystem_ = std::make_unique<MemorySubsystem>(config_);
}

size_t HardwareSimulator::select_optimal_core() const {
    // Select core with least accumulated time (simple load balancing)
    size_t optimal_core = 0;
    double min_time = cpu_cores_[0]->get_metrics().total_execution_time_ms;
    
    for (size_t i = 1; i < cpu_cores_.size(); ++i) {
        double core_time = cpu_cores_[i]->get_metrics().total_execution_time_ms;
        if (core_time < min_time) {
            min_time = core_time;
            optimal_core = i;
        }
    }
    
    return optimal_core;
}

size_t HardwareSimulator::select_optimal_accelerator() const {
    // Select accelerator with least accumulated time
    if (accelerators_.empty()) return 0;
    
    size_t optimal_accelerator = 0;
    double min_time = accelerators_[0]->get_metrics().total_execution_time_ms;
    
    for (size_t i = 1; i < accelerators_.size(); ++i) {
        double acc_time = accelerators_[i]->get_metrics().total_execution_time_ms;
        if (acc_time < min_time) {
            min_time = acc_time;
            optimal_accelerator = i;
        }
    }
    
    return optimal_accelerator;
}

} // namespace HighFidelityAI