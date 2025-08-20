#include "hardware_simulator.hpp"
#include "ai_model.hpp"
#include "profiler.hpp"
#include <iostream>
#include <memory>

using namespace HighFidelityAI;

void run_basic_simulation() {
    std::cout << "=== High-Fidelity AI Model Simulation ===" << std::endl;
    
    // Configure hardware
    HardwareConfig hw_config;
    hw_config.num_cores = 16;
    hw_config.cpu_frequency_ghz = 3.5;
    hw_config.memory_capacity_gb = 64;
    hw_config.memory_bandwidth_gbps = 200.0;
    hw_config.num_accelerators = 4;
    hw_config.accelerator_compute_tflops = 100.0;
    
    std::cout << "\nHardware Configuration:" << std::endl;
    std::cout << "CPU Cores: " << hw_config.num_cores << std::endl;
    std::cout << "CPU Frequency: " << hw_config.cpu_frequency_ghz << " GHz" << std::endl;
    std::cout << "Memory: " << hw_config.memory_capacity_gb << " GB" << std::endl;
    std::cout << "Memory Bandwidth: " << hw_config.memory_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "Accelerators: " << hw_config.num_accelerators << std::endl;
    std::cout << "Accelerator Compute: " << hw_config.accelerator_compute_tflops << " TFLOPS" << std::endl;
    
    // Configure AI model (GPT-like)
    ModelConfig model_config;
    model_config.name = "GPT-Medium";
    model_config.num_layers = 24;
    model_config.hidden_size = 1024;
    model_config.num_attention_heads = 16;
    model_config.intermediate_size = 4096;
    model_config.max_sequence_length = 2048;
    model_config.batch_size = 8;
    
    // Initialize simulators
    HardwareSimulator hw_sim(hw_config);
    AIModelSimulator ai_model(model_config);
    
    ai_model.print_model_summary();
    
    // Run inference simulation
    std::cout << "\n=== Running Inference Simulation ===" << std::endl;
    double inference_time = ai_model.simulate_inference(hw_sim, 1024);
    
    std::cout << "Inference Time: " << inference_time << " ms" << std::endl;
    
    // Print detailed metrics
    hw_sim.print_detailed_metrics();
    
    // Analyze bottlenecks and suggest optimizations
    auto bottlenecks = hw_sim.analyze_bottlenecks();
    if (!bottlenecks.empty()) {
        std::cout << "\nIdentified Bottlenecks:" << std::endl;
        for (const auto& bottleneck : bottlenecks) {
            std::cout << "- " << bottleneck << std::endl;
        }
    }
    
    auto hw_optimizations = hw_sim.suggest_optimizations();
    std::cout << "\nHardware Optimization Suggestions:" << std::endl;
    for (const auto& suggestion : hw_optimizations) {
        std::cout << "- " << suggestion << std::endl;
    }
    
    auto model_optimizations = ai_model.suggest_model_optimizations();
    std::cout << "\nModel Optimization Suggestions:" << std::endl;
    for (const auto& suggestion : model_optimizations) {
        std::cout << "- " << suggestion << std::endl;
    }
}

void demonstrate_hardware_scaling() {
    std::cout << "\n\n=== Hardware Scaling Analysis ===" << std::endl;
    
    ModelConfig model_config;
    model_config.name = "GPT-Large";
    model_config.num_layers = 48;
    model_config.hidden_size = 1536;
    model_config.num_attention_heads = 24;
    model_config.intermediate_size = 6144;
    model_config.max_sequence_length = 2048;
    model_config.batch_size = 4;
    
    AIModelSimulator ai_model(model_config);
    
    // Test different hardware configurations
    std::vector<size_t> core_counts = {4, 8, 16, 32};
    std::vector<size_t> accelerator_counts = {1, 2, 4, 8};
    
    std::cout << "\nCore Count vs Performance:" << std::endl;
    std::cout << "Cores\tTime(ms)\tSpeedup" << std::endl;
    double baseline_time = 0.0;
    
    for (size_t cores : core_counts) {
        HardwareConfig hw_config;
        hw_config.num_cores = cores;
        hw_config.cpu_frequency_ghz = 3.2;
        hw_config.memory_bandwidth_gbps = 150.0;
        hw_config.num_accelerators = 2;
        hw_config.accelerator_compute_tflops = 80.0;
        
        HardwareSimulator hw_sim(hw_config);
        double time = ai_model.simulate_inference(hw_sim, 1024);
        
        if (baseline_time == 0.0) baseline_time = time;
        double speedup = baseline_time / time;
        
        std::cout << cores << "\t" << time << "\t" << speedup << std::endl;
        
        hw_sim.reset_all_metrics();
    }
    
    std::cout << "\nAccelerator Count vs Performance:" << std::endl;
    std::cout << "Accelerators\tTime(ms)\tSpeedup" << std::endl;
    baseline_time = 0.0;
    
    for (size_t accelerators : accelerator_counts) {
        HardwareConfig hw_config;
        hw_config.num_cores = 16;
        hw_config.cpu_frequency_ghz = 3.2;
        hw_config.memory_bandwidth_gbps = 150.0;
        hw_config.num_accelerators = accelerators;
        hw_config.accelerator_compute_tflops = 80.0;
        
        HardwareSimulator hw_sim(hw_config);
        double time = ai_model.simulate_inference(hw_sim, 1024);
        
        if (baseline_time == 0.0) baseline_time = time;
        double speedup = baseline_time / time;
        
        std::cout << accelerators << "\t" << time << "\t" << speedup << std::endl;
        
        hw_sim.reset_all_metrics();
    }
}

void demonstrate_model_scaling() {
    std::cout << "\n\n=== Model Scaling Analysis ===" << std::endl;
    
    HardwareConfig hw_config;
    hw_config.num_cores = 16;
    hw_config.cpu_frequency_ghz = 3.2;
    hw_config.memory_bandwidth_gbps = 200.0;
    hw_config.num_accelerators = 4;
    hw_config.accelerator_compute_tflops = 100.0;
    
    HardwareSimulator hw_sim(hw_config);
    
    std::vector<size_t> layer_counts = {6, 12, 24, 48};
    std::vector<size_t> hidden_sizes = {512, 768, 1024, 1536};
    
    std::cout << "\nLayers vs Performance:" << std::endl;
    std::cout << "Layers\tParams(M)\tTime(ms)\tFLOPS/Token" << std::endl;
    
    for (size_t layers : layer_counts) {
        ModelConfig model_config;
        model_config.num_layers = layers;
        model_config.hidden_size = 768;
        model_config.num_attention_heads = 12;
        model_config.intermediate_size = 3072;
        model_config.max_sequence_length = 1024;
        model_config.batch_size = 1;
        
        AIModelSimulator ai_model(model_config);
        double time = ai_model.simulate_inference(hw_sim, 512);
        size_t params = ai_model.get_total_parameters() / 1000000;  // In millions
        size_t flops_per_token = ai_model.get_total_flops_per_token();
        
        std::cout << layers << "\t" << params << "\t" << time 
                  << "\t" << flops_per_token << std::endl;
        
        hw_sim.reset_all_metrics();
    }
    
    std::cout << "\nHidden Size vs Performance:" << std::endl;
    std::cout << "Hidden\tParams(M)\tTime(ms)\tFLOPS/Token" << std::endl;
    
    for (size_t hidden_size : hidden_sizes) {
        ModelConfig model_config;
        model_config.num_layers = 12;
        model_config.hidden_size = hidden_size;
        model_config.num_attention_heads = hidden_size / 64;  // Standard ratio
        model_config.intermediate_size = hidden_size * 4;
        model_config.max_sequence_length = 1024;
        model_config.batch_size = 1;
        
        AIModelSimulator ai_model(model_config);
        double time = ai_model.simulate_inference(hw_sim, 512);
        size_t params = ai_model.get_total_parameters() / 1000000;
        size_t flops_per_token = ai_model.get_total_flops_per_token();
        
        std::cout << hidden_size << "\t" << params << "\t" << time 
                  << "\t" << flops_per_token << std::endl;
        
        hw_sim.reset_all_metrics();
    }
}

int main() {
    try {
        run_basic_simulation();
        demonstrate_hardware_scaling();
        demonstrate_model_scaling();
        
        std::cout << "\n=== Simulation Complete ===" << std::endl;
        std::cout << "This high-fidelity simulation provides insights for:" << std::endl;
        std::cout << "- Hardware-software co-design optimization" << std::endl;
        std::cout << "- Performance bottleneck identification" << std::endl;
        std::cout << "- Scalability analysis across different configurations" << std::endl;
        std::cout << "- Data-driven recommendations for system design" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}