#include "hardware_simulator.hpp"
#include "ai_model.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace HighFidelityAI;

/**
 * Example: Co-design optimization for different hardware configurations
 */
void hardware_codesign_example() {
    std::cout << "=== Hardware Co-Design Optimization Example ===" << std::endl;
    
    // Define model configuration
    ModelConfig model_config;
    model_config.name = "Efficient-GPT";
    model_config.num_layers = 12;
    model_config.hidden_size = 768;
    model_config.num_attention_heads = 12;
    model_config.intermediate_size = 3072;
    model_config.max_sequence_length = 1024;
    model_config.batch_size = 4;
    
    AIModelSimulator model(model_config);
    
    // Test different hardware configurations
    struct HardwareConfig {
        std::string name;
        size_t cores;
        double frequency;
        size_t memory_gb;
        double bandwidth_gbps;
        size_t accelerators;
        double tflops;
    };
    
    std::vector<HardwareConfig> configs = {
        {"Budget", 8, 2.8, 32, 100.0, 1, 20.0},
        {"Balanced", 16, 3.2, 64, 200.0, 2, 50.0},
        {"High-End", 32, 3.8, 128, 400.0, 4, 100.0},
        {"Extreme", 64, 4.2, 256, 800.0, 8, 200.0}
    };
    
    std::cout << "Hardware\tTime(s)\tEnergy(J)\tCost/Perf\tBottleneck" << std::endl;
    std::cout << "--------\t-------\t---------\t---------\t----------" << std::endl;
    
    for (const auto& config : configs) {
        HighFidelityAI::HardwareConfig hw_config;
        hw_config.num_cores = config.cores;
        hw_config.cpu_frequency_ghz = config.frequency;
        hw_config.memory_capacity_gb = config.memory_gb;
        hw_config.memory_bandwidth_gbps = config.bandwidth_gbps;
        hw_config.num_accelerators = config.accelerators;
        hw_config.accelerator_compute_tflops = config.tflops;
        
        HardwareSimulator hw_sim(hw_config);
        
        double time_ms = model.simulate_inference(hw_sim, 512);
        double time_s = time_ms / 1000.0;
        auto metrics = hw_sim.get_aggregate_metrics();
        double energy_j = metrics.energy_consumption_j;
        
        // Simple cost/performance metric
        double cost_perf = energy_j * time_s / 1000.0;
        
        // Identify primary bottleneck
        std::string bottleneck = "Compute";
        if (metrics.memory_time_ms > metrics.compute_time_ms) {
            bottleneck = "Memory";
        }
        if (metrics.utilization_cpu < 50.0) {
            bottleneck = "Underutil";
        }
        
        std::cout << config.name << "\t\t" 
                  << std::fixed << std::setprecision(2) << time_s << "\t"
                  << std::setprecision(0) << energy_j << "\t"
                  << std::setprecision(2) << cost_perf << "\t\t"
                  << bottleneck << std::endl;
        
        hw_sim.reset_all_metrics();
    }
    
    std::cout << "\nInsights:" << std::endl;
    std::cout << "- Compare Time vs Energy vs Cost for different configurations" << std::endl;
    std::cout << "- Identify optimal hardware for your specific workload" << std::endl;
    std::cout << "- Understand scaling characteristics and bottlenecks" << std::endl;
}

/**
 * Example: Model architecture exploration
 */
void model_architecture_exploration() {
    std::cout << "\n=== Model Architecture Exploration Example ===" << std::endl;
    
    HighFidelityAI::HardwareConfig hw_config;
    hw_config.num_cores = 16;
    hw_config.cpu_frequency_ghz = 3.5;
    hw_config.memory_capacity_gb = 64;
    hw_config.memory_bandwidth_gbps = 200.0;
    hw_config.num_accelerators = 4;
    hw_config.accelerator_compute_tflops = 80.0;
    
    HardwareSimulator hw_sim(hw_config);
    
    struct ModelVariant {
        std::string name;
        size_t layers;
        size_t hidden_size;
        size_t heads;
        size_t seq_len;
    };
    
    std::vector<ModelVariant> variants = {
        {"Small", 6, 512, 8, 512},
        {"Base", 12, 768, 12, 1024},
        {"Large", 24, 1024, 16, 1024},
        {"XL", 48, 1280, 20, 2048}
    };
    
    std::cout << "Model\tParams(M)\tFLOPs/Tok\tTime(ms)\tMem(GB)\tEfficiency" << std::endl;
    std::cout << "-----\t---------\t---------\t--------\t-------\t----------" << std::endl;
    
    for (const auto& variant : variants) {
        ModelConfig model_config;
        model_config.name = variant.name;
        model_config.num_layers = variant.layers;
        model_config.hidden_size = variant.hidden_size;
        model_config.num_attention_heads = variant.heads;
        model_config.intermediate_size = variant.hidden_size * 4;
        model_config.max_sequence_length = variant.seq_len;
        model_config.batch_size = 1;
        
        AIModelSimulator model(model_config);
        
        double time_ms = model.simulate_inference(hw_sim, variant.seq_len / 2);
        size_t params = model.get_total_parameters();
        size_t flops_per_token = model.get_total_flops_per_token();
        
        // Estimate memory usage (parameters + activations)
        double memory_gb = (params * 4.0 + variant.seq_len * variant.hidden_size * 4.0) / (1024.0 * 1024.0 * 1024.0);
        
        // Simple efficiency metric (FLOPS per ms)
        double efficiency = flops_per_token / (time_ms * 1000.0);
        
        std::cout << variant.name << "\t" 
                  << (params / 1000000) << "\t\t"
                  << (flops_per_token / 1000000) << "M\t"
                  << std::fixed << std::setprecision(1) << time_ms << "\t"
                  << std::setprecision(2) << memory_gb << "\t"
                  << std::scientific << std::setprecision(2) << efficiency << std::endl;
        
        hw_sim.reset_all_metrics();
    }
    
    std::cout << "\nInsights:" << std::endl;
    std::cout << "- Understand parameter vs performance trade-offs" << std::endl;
    std::cout << "- Identify memory constraints for different model sizes" << std::endl;
    std::cout << "- Choose optimal architecture for your hardware budget" << std::endl;
}

/**
 * Example: Optimization recommendations
 */
void optimization_recommendations_example() {
    std::cout << "\n=== Optimization Recommendations Example ===" << std::endl;
    
    // Create a configuration with some inefficiencies
    HighFidelityAI::HardwareConfig hw_config;
    hw_config.num_cores = 4;  // Low core count
    hw_config.cpu_frequency_ghz = 2.0;  // Low frequency
    hw_config.memory_capacity_gb = 16;  // Low memory
    hw_config.memory_bandwidth_gbps = 50.0;  // Low bandwidth
    hw_config.num_accelerators = 1;  // Single accelerator
    hw_config.accelerator_compute_tflops = 10.0;  // Low compute
    
    HardwareSimulator hw_sim(hw_config);
    
    ModelConfig model_config;
    model_config.name = "Test-Model";
    model_config.num_layers = 24;  // Large model
    model_config.hidden_size = 1024;
    model_config.num_attention_heads = 16;
    model_config.intermediate_size = 4096;
    model_config.max_sequence_length = 2048;
    model_config.batch_size = 8;  // Large batch
    
    AIModelSimulator model(model_config);
    
    std::cout << "Configuration with intentional bottlenecks:" << std::endl;
    std::cout << "- Low-end hardware with large model" << std::endl;
    
    double time_ms = model.simulate_inference(hw_sim, 1024);
    auto metrics = hw_sim.get_aggregate_metrics();
    
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "Execution Time: " << (time_ms / 1000.0) << " seconds" << std::endl;
    std::cout << "Energy Usage: " << metrics.energy_consumption_j << " J" << std::endl;
    std::cout << "Memory Utilization: " << (metrics.memory_accesses > 0 ? "High" : "Low") << std::endl;
    
    std::cout << "\nIdentified Bottlenecks:" << std::endl;
    auto bottlenecks = hw_sim.analyze_bottlenecks();
    for (const auto& bottleneck : bottlenecks) {
        std::cout << "⚠️  " << bottleneck << std::endl;
    }
    
    std::cout << "\nHardware Optimization Suggestions:" << std::endl;
    auto hw_suggestions = hw_sim.suggest_optimizations();
    for (size_t i = 0; i < hw_suggestions.size() && i < 3; ++i) {
        std::cout << "🔧 " << hw_suggestions[i] << std::endl;
    }
    
    std::cout << "\nModel Optimization Suggestions:" << std::endl;
    auto model_suggestions = model.suggest_model_optimizations();
    for (size_t i = 0; i < model_suggestions.size() && i < 3; ++i) {
        std::cout << "⚡ " << model_suggestions[i] << std::endl;
    }
}

int main() {
    try {
        hardware_codesign_example();
        model_architecture_exploration();
        optimization_recommendations_example();
        
        std::cout << "\n=== Examples Complete ===" << std::endl;
        std::cout << "These examples demonstrate the simulator's ability to:" << std::endl;
        std::cout << "✅ Compare hardware configurations objectively" << std::endl;
        std::cout << "✅ Analyze model architecture trade-offs" << std::endl;
        std::cout << "✅ Provide actionable optimization recommendations" << std::endl;
        std::cout << "✅ Enable data-driven hardware-software co-design decisions" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}