#include "ai_model.hpp"
#include "hardware_simulator.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace HighFidelityAI {

// Helper functions for FLOPS calculation
size_t ModelLayer::calculate_matmul_flops(size_t m, size_t n, size_t k) const {
    return 2 * m * n * k;  // 2 FLOPS per multiply-add
}

size_t ModelLayer::calculate_attention_flops() const {
    size_t seq_len = config_.sequence_length;
    size_t hidden_size = config_.hidden_size;
    size_t batch_size = config_.batch_size;
    
    // Q, K, V projections
    size_t projection_flops = 3 * calculate_matmul_flops(batch_size * seq_len, hidden_size, hidden_size);
    
    // Attention scores: Q * K^T
    size_t attention_flops = calculate_matmul_flops(batch_size * config_.num_heads, seq_len, seq_len);
    attention_flops *= config_.head_dimension;  // For each head
    
    // Attention weights * V
    size_t output_flops = calculate_matmul_flops(batch_size * config_.num_heads, seq_len, config_.head_dimension);
    output_flops *= seq_len;
    
    // Output projection
    size_t final_projection_flops = calculate_matmul_flops(batch_size * seq_len, hidden_size, hidden_size);
    
    return projection_flops + attention_flops + output_flops + final_projection_flops;
}

// AttentionLayer Implementation
AttentionLayer::AttentionLayer(const LayerConfig& config) : ModelLayer(config) {
    initialize_operations();
}

void AttentionLayer::initialize_operations() {
    compute_ops_.clear();
    memory_ops_.clear();
    
    size_t seq_len = config_.sequence_length;
    size_t hidden_size = config_.hidden_size;
    size_t batch_size = config_.batch_size;
    
    // Q, K, V projections
    size_t qkv_flops = 3 * calculate_matmul_flops(batch_size * seq_len, hidden_size, hidden_size);
    compute_ops_.emplace_back(OperationType::MATRIX_MULTIPLY, "QKV_Projection", qkv_flops);
    compute_ops_.back().is_parallelizable = true;
    compute_ops_.back().parallelism_factor = config_.num_heads;
    
    // Attention computation
    size_t attention_flops = calculate_attention_flops() - qkv_flops;
    compute_ops_.emplace_back(OperationType::ATTENTION, "Attention_Computation", attention_flops);
    compute_ops_.back().is_memory_bound = true;
    compute_ops_.back().arithmetic_intensity = 0.5;  // Memory intensive
    
    // Softmax activation
    size_t softmax_flops = batch_size * config_.num_heads * seq_len * seq_len * 10;  // Approximation
    compute_ops_.emplace_back(OperationType::ACTIVATION, "Softmax", softmax_flops);
    
    // Output projection
    size_t output_flops = calculate_matmul_flops(batch_size * seq_len, hidden_size, hidden_size);
    compute_ops_.emplace_back(OperationType::MATRIX_MULTIPLY, "Output_Projection", output_flops);
    
    // Memory operations
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t weight_size = 4 * hidden_size * hidden_size * sizeof(float);  // Q, K, V, O weights
    size_t intermediate_size = batch_size * config_.num_heads * seq_len * seq_len * sizeof(float);
    
    memory_ops_.emplace_back("Load_Input", input_size, false);
    memory_ops_.emplace_back("Load_Weights", weight_size, false);
    memory_ops_.emplace_back("Store_Intermediate", intermediate_size, true);
    memory_ops_.emplace_back("Store_Output", input_size, true);
}

std::vector<ComputeOperation> AttentionLayer::get_compute_operations() const {
    return compute_ops_;
}

std::vector<MemoryOperation> AttentionLayer::get_memory_operations() const {
    return memory_ops_;
}

double AttentionLayer::simulate_forward_pass(HardwareSimulator& hw_sim) {
    double total_time = 0.0;
    
    // Execute memory operations first
    for (const auto& mem_op : memory_ops_) {
        if (!mem_op.is_write) {  // Load operations
            total_time += hw_sim.simulate_memory_operation(mem_op);
        }
    }
    
    // Execute compute operations
    for (const auto& compute_op : compute_ops_) {
        if (compute_op.is_parallelizable && compute_op.type == OperationType::MATRIX_MULTIPLY) {
            total_time += hw_sim.simulate_parallel_operation(compute_op);
        } else {
            total_time += hw_sim.simulate_compute_operation(compute_op);
        }
    }
    
    // Execute store operations
    for (const auto& mem_op : memory_ops_) {
        if (mem_op.is_write) {  // Store operations
            total_time += hw_sim.simulate_memory_operation(mem_op);
        }
    }
    
    return total_time;
}

// FeedforwardLayer Implementation
FeedforwardLayer::FeedforwardLayer(const LayerConfig& config) : ModelLayer(config) {
    initialize_operations();
}

void FeedforwardLayer::initialize_operations() {
    compute_ops_.clear();
    memory_ops_.clear();
    
    size_t seq_len = config_.sequence_length;
    size_t hidden_size = config_.hidden_size;
    size_t intermediate_size = config_.intermediate_size;
    size_t batch_size = config_.batch_size;
    
    // First linear transformation
    size_t linear1_flops = calculate_matmul_flops(batch_size * seq_len, hidden_size, intermediate_size);
    compute_ops_.emplace_back(OperationType::MATRIX_MULTIPLY, "Linear1", linear1_flops);
    compute_ops_.back().is_parallelizable = true;
    
    // Activation function (GELU/ReLU)
    size_t activation_flops = batch_size * seq_len * intermediate_size * 5;  // Approximation for GELU
    compute_ops_.emplace_back(OperationType::ACTIVATION, "Activation", activation_flops);
    
    // Second linear transformation
    size_t linear2_flops = calculate_matmul_flops(batch_size * seq_len, intermediate_size, hidden_size);
    compute_ops_.emplace_back(OperationType::MATRIX_MULTIPLY, "Linear2", linear2_flops);
    compute_ops_.back().is_parallelizable = true;
    
    // Memory operations
    size_t input_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t weight1_size = hidden_size * intermediate_size * sizeof(float);
    size_t weight2_size = intermediate_size * hidden_size * sizeof(float);
    size_t intermediate_data_size = batch_size * seq_len * intermediate_size * sizeof(float);
    
    memory_ops_.emplace_back("Load_Input", input_size, false);
    memory_ops_.emplace_back("Load_Weight1", weight1_size, false);
    memory_ops_.emplace_back("Load_Weight2", weight2_size, false);
    memory_ops_.emplace_back("Store_Intermediate", intermediate_data_size, true);
    memory_ops_.emplace_back("Store_Output", input_size, true);
}

std::vector<ComputeOperation> FeedforwardLayer::get_compute_operations() const {
    return compute_ops_;
}

std::vector<MemoryOperation> FeedforwardLayer::get_memory_operations() const {
    return memory_ops_;
}

double FeedforwardLayer::simulate_forward_pass(HardwareSimulator& hw_sim) {
    double total_time = 0.0;
    
    // Load operations
    for (const auto& mem_op : memory_ops_) {
        if (!mem_op.is_write) {
            total_time += hw_sim.simulate_memory_operation(mem_op);
        }
    }
    
    // Compute operations - all can be executed on accelerators
    for (const auto& compute_op : compute_ops_) {
        if (compute_op.is_parallelizable) {
            total_time += hw_sim.simulate_parallel_operation(compute_op);
        } else {
            total_time += hw_sim.simulate_compute_operation(compute_op);
        }
    }
    
    // Store operations
    for (const auto& mem_op : memory_ops_) {
        if (mem_op.is_write) {
            total_time += hw_sim.simulate_memory_operation(mem_op);
        }
    }
    
    return total_time;
}

// EmbeddingLayer Implementation
EmbeddingLayer::EmbeddingLayer(const LayerConfig& config) : ModelLayer(config) {
    initialize_operations();
}

void EmbeddingLayer::initialize_operations() {
    compute_ops_.clear();
    memory_ops_.clear();
    
    size_t seq_len = config_.sequence_length;
    size_t hidden_size = config_.hidden_size;
    size_t batch_size = config_.batch_size;
    
    // Embedding lookup (essentially a gather operation)
    // No actual compute, mostly memory intensive
    size_t lookup_ops = batch_size * seq_len;
    compute_ops_.emplace_back(OperationType::MEMORY_COPY, "Embedding_Lookup", lookup_ops * 10);
    compute_ops_.back().is_memory_bound = true;
    compute_ops_.back().arithmetic_intensity = 0.1;  // Very memory intensive
    
    // Positional encoding addition
    size_t pos_encoding_flops = batch_size * seq_len * hidden_size;
    compute_ops_.emplace_back(OperationType::ACTIVATION, "Positional_Encoding", pos_encoding_flops);
    
    // Memory operations
    size_t vocab_size = 50000;  // Typical vocabulary size
    size_t embedding_table_size = vocab_size * hidden_size * sizeof(float);
    size_t output_size = batch_size * seq_len * hidden_size * sizeof(float);
    size_t input_tokens_size = batch_size * seq_len * sizeof(int);
    
    memory_ops_.emplace_back("Load_Input_Tokens", input_tokens_size, false);
    memory_ops_.emplace_back("Load_Embedding_Table", embedding_table_size, false);
    memory_ops_.emplace_back("Store_Embeddings", output_size, true);
}

std::vector<ComputeOperation> EmbeddingLayer::get_compute_operations() const {
    return compute_ops_;
}

std::vector<MemoryOperation> EmbeddingLayer::get_memory_operations() const {
    return memory_ops_;
}

double EmbeddingLayer::simulate_forward_pass(HardwareSimulator& hw_sim) {
    double total_time = 0.0;
    
    // Load operations
    for (const auto& mem_op : memory_ops_) {
        if (!mem_op.is_write) {
            total_time += hw_sim.simulate_memory_operation(mem_op);
        }
    }
    
    // Compute operations
    for (const auto& compute_op : compute_ops_) {
        total_time += hw_sim.simulate_compute_operation(compute_op);
    }
    
    // Store operations
    for (const auto& mem_op : memory_ops_) {
        if (mem_op.is_write) {
            total_time += hw_sim.simulate_memory_operation(mem_op);
        }
    }
    
    return total_time;
}

// AIModelSimulator Implementation
AIModelSimulator::AIModelSimulator(const ModelConfig& config) : config_(config) {
    build_model();
}

void AIModelSimulator::set_config(const ModelConfig& config) {
    config_ = config;
    model_built_ = false;
    layers_.clear();
    build_model();
}

void AIModelSimulator::build_model() {
    if (model_built_) return;
    
    layers_.clear();
    
    // Add embedding layer
    layers_.push_back(std::make_unique<EmbeddingLayer>(create_embedding_config()));
    
    // Add transformer layers
    create_transformer_layers();
    
    model_built_ = true;
}

void AIModelSimulator::add_layer(std::unique_ptr<ModelLayer> layer) {
    layers_.push_back(std::move(layer));
}

double AIModelSimulator::simulate_inference(HardwareSimulator& hw_sim, size_t input_tokens) {
    if (!model_built_) {
        build_model();
    }
    
    if (input_tokens == 0) {
        input_tokens = config_.max_sequence_length;
    }
    
    double total_time = 0.0;
    
    // Execute all layers sequentially
    for (auto& layer : layers_) {
        double layer_time = layer->simulate_forward_pass(hw_sim);
        total_time += layer_time;
    }
    
    return total_time;
}

double AIModelSimulator::simulate_training_step(HardwareSimulator& hw_sim, size_t input_tokens) {
    // Training requires both forward and backward passes
    double forward_time = simulate_inference(hw_sim, input_tokens);
    
    // Backward pass typically takes 2x forward pass time
    double backward_time = forward_time * 2.0;
    
    // Parameter updates
    double update_time = forward_time * 0.1;
    
    return forward_time + backward_time + update_time;
}

std::vector<ComputeOperation> AIModelSimulator::get_all_compute_operations() const {
    std::vector<ComputeOperation> all_ops;
    
    for (const auto& layer : layers_) {
        auto layer_ops = layer->get_compute_operations();
        all_ops.insert(all_ops.end(), layer_ops.begin(), layer_ops.end());
    }
    
    return all_ops;
}

std::vector<MemoryOperation> AIModelSimulator::get_all_memory_operations() const {
    std::vector<MemoryOperation> all_ops;
    
    for (const auto& layer : layers_) {
        auto layer_ops = layer->get_memory_operations();
        all_ops.insert(all_ops.end(), layer_ops.begin(), layer_ops.end());
    }
    
    return all_ops;
}

size_t AIModelSimulator::get_total_parameters() const {
    size_t total_params = 0;
    
    // Embedding parameters
    total_params += config_.vocab_size * config_.hidden_size;
    
    // Transformer layer parameters
    for (size_t i = 0; i < config_.num_layers; ++i) {
        // Attention parameters: Q, K, V, O projections
        total_params += 4 * config_.hidden_size * config_.hidden_size;
        
        // Feedforward parameters
        total_params += 2 * config_.hidden_size * config_.intermediate_size;
        
        // Layer norm parameters
        if (config_.use_layer_norm) {
            total_params += 2 * config_.hidden_size;  // gamma, beta
        }
    }
    
    return total_params;
}

size_t AIModelSimulator::get_total_flops_per_token() const {
    size_t total_flops = 0;
    
    auto compute_ops = get_all_compute_operations();
    for (const auto& op : compute_ops) {
        total_flops += op.flops;
    }
    
    return total_flops / config_.max_sequence_length;
}

std::vector<std::string> AIModelSimulator::analyze_computational_bottlenecks() const {
    std::vector<std::string> bottlenecks;
    
    auto compute_ops = get_all_compute_operations();
    size_t total_flops = 0;
    size_t attention_flops = 0;
    size_t matmul_flops = 0;
    
    for (const auto& op : compute_ops) {
        total_flops += op.flops;
        if (op.type == OperationType::ATTENTION) {
            attention_flops += op.flops;
        } else if (op.type == OperationType::MATRIX_MULTIPLY) {
            matmul_flops += op.flops;
        }
    }
    
    if (attention_flops > total_flops * 0.4) {
        bottlenecks.push_back("Attention operations dominate compute (consider attention optimization)");
    }
    
    if (matmul_flops > total_flops * 0.6) {
        bottlenecks.push_back("Matrix multiplications dominate compute (optimize GEMM operations)");
    }
    
    if (config_.num_layers > 24) {
        bottlenecks.push_back("Very deep model may have gradient flow issues");
    }
    
    if (config_.hidden_size > 4096) {
        bottlenecks.push_back("Large hidden size may be memory-bandwidth limited");
    }
    
    return bottlenecks;
}

std::vector<std::string> AIModelSimulator::suggest_model_optimizations() const {
    std::vector<std::string> suggestions;
    
    suggestions.push_back("Consider attention optimization techniques (Flash Attention, etc.)");
    suggestions.push_back("Use mixed-precision training to reduce memory bandwidth");
    suggestions.push_back("Implement gradient checkpointing to trade compute for memory");
    suggestions.push_back("Consider model parallelism for large models");
    suggestions.push_back("Optimize attention patterns (sparse attention, local attention)");
    suggestions.push_back("Use efficient activation functions");
    suggestions.push_back("Consider knowledge distillation for smaller models");
    
    return suggestions;
}

void AIModelSimulator::print_model_summary() const {
    std::cout << "\n=== AI Model Summary ===" << std::endl;
    std::cout << "Model Name: " << config_.name << std::endl;
    std::cout << "Number of Layers: " << config_.num_layers << std::endl;
    std::cout << "Hidden Size: " << config_.hidden_size << std::endl;
    std::cout << "Attention Heads: " << config_.num_attention_heads << std::endl;
    std::cout << "Intermediate Size: " << config_.intermediate_size << std::endl;
    std::cout << "Vocabulary Size: " << config_.vocab_size << std::endl;
    std::cout << "Max Sequence Length: " << config_.max_sequence_length << std::endl;
    std::cout << "Batch Size: " << config_.batch_size << std::endl;
    std::cout << "Total Parameters: " << get_total_parameters() << std::endl;
    std::cout << "FLOPS per Token: " << get_total_flops_per_token() << std::endl;
    
    auto bottlenecks = analyze_computational_bottlenecks();
    if (!bottlenecks.empty()) {
        std::cout << "\nComputational Bottlenecks:" << std::endl;
        for (const auto& bottleneck : bottlenecks) {
            std::cout << "- " << bottleneck << std::endl;
        }
    }
}

void AIModelSimulator::create_transformer_layers() {
    for (size_t i = 0; i < config_.num_layers; ++i) {
        // Attention layer
        layers_.push_back(std::make_unique<AttentionLayer>(create_attention_config()));
        
        // Feedforward layer
        layers_.push_back(std::make_unique<FeedforwardLayer>(create_feedforward_config()));
    }
}

LayerConfig AIModelSimulator::create_attention_config() const {
    LayerConfig config;
    config.name = "MultiHeadAttention";
    config.type = "attention";
    config.hidden_size = config_.hidden_size;
    config.sequence_length = config_.max_sequence_length;
    config.batch_size = config_.batch_size;
    config.num_heads = config_.num_attention_heads;
    config.head_dimension = config_.hidden_size / config_.num_attention_heads;
    return config;
}

LayerConfig AIModelSimulator::create_feedforward_config() const {
    LayerConfig config;
    config.name = "FeedForward";
    config.type = "feedforward";
    config.hidden_size = config_.hidden_size;
    config.sequence_length = config_.max_sequence_length;
    config.batch_size = config_.batch_size;
    config.intermediate_size = config_.intermediate_size;
    return config;
}

LayerConfig AIModelSimulator::create_embedding_config() const {
    LayerConfig config;
    config.name = "Embedding";
    config.type = "embedding";
    config.hidden_size = config_.hidden_size;
    config.sequence_length = config_.max_sequence_length;
    config.batch_size = config_.batch_size;
    return config;
}

} // namespace HighFidelityAI