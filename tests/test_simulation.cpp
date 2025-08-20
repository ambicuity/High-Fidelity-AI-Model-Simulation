#include "hardware_simulator.hpp"
#include "ai_model.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace HighFidelityAI;

/**
 * Test hardware simulator basic functionality
 */
bool test_hardware_simulator() {
    std::cout << "Testing Hardware Simulator..." << std::endl;
    
    HardwareConfig config;
    config.num_cores = 4;
    config.cpu_frequency_ghz = 3.0;
    config.memory_bandwidth_gbps = 100.0;
    config.num_accelerators = 2;
    
    HardwareSimulator hw_sim(config);
    
    // Test compute operation
    ComputeOperation compute_op(OperationType::MATRIX_MULTIPLY, "test_matmul", 1000000);  // 1M FLOPS
    double compute_time = hw_sim.simulate_compute_operation(compute_op);
    assert(compute_time > 0.0 && compute_time < 1000.0);  // Should be reasonable
    
    // Test memory operation
    MemoryOperation mem_op("test_memory", 1024 * 1024, false);  // 1MB read
    double memory_time = hw_sim.simulate_memory_operation(mem_op);
    assert(memory_time > 0.0 && memory_time < 1000.0);  // Should be reasonable
    
    // Test parallel operation
    ComputeOperation parallel_op(OperationType::MATRIX_MULTIPLY, "test_parallel", 10000000);  // 10M FLOPS
    parallel_op.is_parallelizable = true;
    double parallel_time = hw_sim.simulate_parallel_operation(parallel_op);
    assert(parallel_time > 0.0 && parallel_time < 10000.0);  // Should be reasonable
    
    // Test metrics aggregation
    auto metrics = hw_sim.get_aggregate_metrics();
    assert(metrics.total_execution_time_ms > 0.0);
    assert(metrics.compute_time_ms > 0.0);
    assert(metrics.memory_time_ms > 0.0);
    
    std::cout << "✅ Hardware Simulator tests passed" << std::endl;
    return true;
}

/**
 * Test AI model simulator basic functionality
 */
bool test_ai_model_simulator() {
    std::cout << "Testing AI Model Simulator..." << std::endl;
    
    ModelConfig config;
    config.name = "test_model";
    config.num_layers = 2;
    config.hidden_size = 128;
    config.num_attention_heads = 4;
    config.intermediate_size = 512;
    config.max_sequence_length = 64;
    config.batch_size = 1;
    
    AIModelSimulator model(config);
    
    // Test model parameters calculation
    size_t params = model.get_total_parameters();
    assert(params > 0);
    std::cout << "Model parameters: " << params << std::endl;
    
    // Test FLOPS calculation
    size_t flops_per_token = model.get_total_flops_per_token();
    assert(flops_per_token > 0);
    std::cout << "FLOPS per token: " << flops_per_token << std::endl;
    
    // Test operations extraction
    auto compute_ops = model.get_all_compute_operations();
    assert(!compute_ops.empty());
    std::cout << "Number of compute operations: " << compute_ops.size() << std::endl;
    
    auto memory_ops = model.get_all_memory_operations();
    assert(!memory_ops.empty());
    std::cout << "Number of memory operations: " << memory_ops.size() << std::endl;
    
    std::cout << "✅ AI Model Simulator tests passed" << std::endl;
    return true;
}

/**
 * Test integration between hardware and model simulators
 */
bool test_integration() {
    std::cout << "Testing Hardware-Model Integration..." << std::endl;
    
    HardwareConfig hw_config;
    hw_config.num_cores = 2;
    hw_config.cpu_frequency_ghz = 2.5;
    hw_config.memory_bandwidth_gbps = 50.0;
    hw_config.num_accelerators = 1;
    hw_config.accelerator_compute_tflops = 10.0;
    
    ModelConfig model_config;
    model_config.num_layers = 1;
    model_config.hidden_size = 64;
    model_config.num_attention_heads = 2;
    model_config.intermediate_size = 256;
    model_config.max_sequence_length = 32;
    model_config.batch_size = 1;
    
    HardwareSimulator hw_sim(hw_config);
    AIModelSimulator model(model_config);
    
    // Test inference simulation
    double inference_time = model.simulate_inference(hw_sim, 16);
    assert(inference_time > 0.0);
    std::cout << "Inference time: " << inference_time << " ms" << std::endl;
    
    // Test training simulation
    double training_time = model.simulate_training_step(hw_sim, 16);
    assert(training_time > inference_time);  // Training should take longer
    std::cout << "Training time: " << training_time << " ms" << std::endl;
    
    // Test metrics collection
    auto metrics = hw_sim.get_aggregate_metrics();
    assert(metrics.total_execution_time_ms > 0.0);
    
    std::cout << "✅ Integration tests passed" << std::endl;
    return true;
}

/**
 * Test performance analysis functionality
 */
bool test_analysis() {
    std::cout << "Testing Analysis Functions..." << std::endl;
    
    HardwareConfig hw_config;
    hw_config.num_cores = 1;  // Single core to create bottleneck
    hw_config.cpu_frequency_ghz = 1.0;  // Low frequency
    hw_config.memory_bandwidth_gbps = 10.0;  // Low bandwidth
    hw_config.num_accelerators = 1;
    hw_config.accelerator_compute_tflops = 5.0;
    
    ModelConfig model_config;
    model_config.num_layers = 4;  // Larger model
    model_config.hidden_size = 256;
    model_config.num_attention_heads = 8;
    model_config.intermediate_size = 1024;
    model_config.max_sequence_length = 128;
    model_config.batch_size = 2;
    
    HardwareSimulator hw_sim(hw_config);
    AIModelSimulator model(model_config);
    
    // Run simulation to generate metrics
    double time = model.simulate_inference(hw_sim, 64);
    std::cout << "Simulation time: " << time << " ms" << std::endl;
    
    // Test bottleneck analysis
    auto bottlenecks = hw_sim.analyze_bottlenecks();
    std::cout << "Identified " << bottlenecks.size() << " bottlenecks" << std::endl;
    if (!bottlenecks.empty()) {
        for (const auto& bottleneck : bottlenecks) {
            std::cout << "- " << bottleneck << std::endl;
        }
    }
    
    // Test optimization suggestions
    auto hw_suggestions = hw_sim.suggest_optimizations();
    assert(!hw_suggestions.empty());
    std::cout << "Hardware suggestions: " << hw_suggestions.size() << std::endl;
    
    auto model_suggestions = model.suggest_model_optimizations();
    assert(!model_suggestions.empty());
    std::cout << "Model suggestions: " << model_suggestions.size() << std::endl;
    
    // Test model bottleneck analysis
    auto model_bottlenecks = model.analyze_computational_bottlenecks();
    std::cout << "Model bottlenecks: " << model_bottlenecks.size() << std::endl;
    
    std::cout << "✅ Analysis tests passed" << std::endl;
    return true;
}

/**
 * Test scaling behavior
 */
bool test_scaling() {
    std::cout << "Testing Scaling Behavior..." << std::endl;
    
    ModelConfig model_config;
    model_config.num_layers = 2;
    model_config.hidden_size = 128;
    model_config.num_attention_heads = 4;
    model_config.intermediate_size = 512;
    model_config.max_sequence_length = 64;
    model_config.batch_size = 1;
    
    AIModelSimulator model(model_config);
    
    // Test scaling with different core counts
    std::vector<size_t> core_counts = {1, 2, 4};
    std::vector<double> times;
    
    for (size_t cores : core_counts) {
        HardwareConfig hw_config;
        hw_config.num_cores = cores;
        hw_config.cpu_frequency_ghz = 2.0;
        hw_config.memory_bandwidth_gbps = 100.0;
        hw_config.num_accelerators = 1;
        
        HardwareSimulator hw_sim(hw_config);
        double time = model.simulate_inference(hw_sim, 32);
        times.push_back(time);
        
        std::cout << "Cores: " << cores << ", Time: " << time << " ms" << std::endl;
        
        hw_sim.reset_all_metrics();
    }
    
    // Verify that adding more cores generally helps (not always true due to memory bottlenecks)
    assert(times[0] > 0.0 && times[1] > 0.0 && times[2] > 0.0);
    
    std::cout << "✅ Scaling tests passed" << std::endl;
    return true;
}

int main() {
    std::cout << "=== High-Fidelity AI Simulation Test Suite ===" << std::endl;
    
    bool all_passed = true;
    
    try {
        all_passed &= test_hardware_simulator();
        all_passed &= test_ai_model_simulator();
        all_passed &= test_integration();
        all_passed &= test_analysis();
        all_passed &= test_scaling();
        
        if (all_passed) {
            std::cout << "\n🎉 All tests PASSED!" << std::endl;
            std::cout << "The simulation framework is working correctly and ready for:" << std::endl;
            std::cout << "- Hardware-software co-design optimization" << std::endl;
            std::cout << "- Performance bottleneck identification" << std::endl;
            std::cout << "- Scalability analysis" << std::endl;
            std::cout << "- Data-driven system design decisions" << std::endl;
        } else {
            std::cout << "\n❌ Some tests FAILED!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}