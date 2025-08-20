#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstddef>

namespace HighFidelityAI {

/**
 * Types of operations in AI workloads
 */
enum class OperationType {
    MATRIX_MULTIPLY,
    CONVOLUTION,
    ATTENTION,
    ACTIVATION,
    NORMALIZATION,
    MEMORY_COPY,
    MEMORY_ALLOC
};

/**
 * Compute operation definition
 */
struct ComputeOperation {
    OperationType type;
    std::string name;
    
    // Operation parameters
    size_t input_size = 0;
    size_t output_size = 0;
    size_t weight_size = 0;
    size_t flops = 0;              // Floating point operations
    
    // Parallelization parameters
    bool is_parallelizable = true;
    size_t parallelism_factor = 1;
    
    // Memory access pattern
    bool is_memory_bound = false;
    double arithmetic_intensity = 1.0;  // FLOPS per byte
    
    ComputeOperation(OperationType t, const std::string& n, size_t flops_count)
        : type(t), name(n), flops(flops_count) {}
};

/**
 * Memory operation definition  
 */
struct MemoryOperation {
    std::string name;
    size_t address = 0;
    size_t size = 0;
    bool is_write = false;
    bool is_sequential = true;
    
    MemoryOperation(const std::string& n, size_t sz, bool write = false)
        : name(n), size(sz), is_write(write) {}
};

/**
 * AI model layer configuration
 */
struct LayerConfig {
    std::string name;
    std::string type;  // "attention", "feedforward", "embedding", etc.
    
    // Common parameters
    size_t hidden_size = 512;
    size_t sequence_length = 1024;
    size_t batch_size = 1;
    
    // Attention-specific
    size_t num_heads = 8;
    size_t head_dimension = 64;
    
    // Feedforward-specific  
    size_t intermediate_size = 2048;
    
    // Convolution-specific
    size_t kernel_size = 3;
    size_t stride = 1;
    size_t channels = 256;
};

/**
 * Forward declaration
 */
class HardwareSimulator;

/**
 * Base class for AI model layers
 */
class ModelLayer {
public:
    explicit ModelLayer(const LayerConfig& config) : config_(config) {}
    virtual ~ModelLayer() = default;
    
    virtual std::vector<ComputeOperation> get_compute_operations() const = 0;
    virtual std::vector<MemoryOperation> get_memory_operations() const = 0;
    virtual double simulate_forward_pass(HardwareSimulator& hw_sim) = 0;
    
    const LayerConfig& get_config() const { return config_; }
    
protected:
    LayerConfig config_;
    
    // Helper methods for FLOPS calculation
    size_t calculate_matmul_flops(size_t m, size_t n, size_t k) const;
    size_t calculate_attention_flops() const;
};

/**
 * Multi-head attention layer
 */
class AttentionLayer : public ModelLayer {
public:
    explicit AttentionLayer(const LayerConfig& config);
    
    std::vector<ComputeOperation> get_compute_operations() const override;
    std::vector<MemoryOperation> get_memory_operations() const override;
    double simulate_forward_pass(HardwareSimulator& hw_sim) override;
    
private:
    void initialize_operations();
    
    std::vector<ComputeOperation> compute_ops_;
    std::vector<MemoryOperation> memory_ops_;
};

/**
 * Feedforward layer
 */
class FeedforwardLayer : public ModelLayer {
public:
    explicit FeedforwardLayer(const LayerConfig& config);
    
    std::vector<ComputeOperation> get_compute_operations() const override;
    std::vector<MemoryOperation> get_memory_operations() const override;
    double simulate_forward_pass(HardwareSimulator& hw_sim) override;
    
private:
    void initialize_operations();
    
    std::vector<ComputeOperation> compute_ops_;
    std::vector<MemoryOperation> memory_ops_;
};

/**
 * Embedding layer
 */
class EmbeddingLayer : public ModelLayer {
public:
    explicit EmbeddingLayer(const LayerConfig& config);
    
    std::vector<ComputeOperation> get_compute_operations() const override;
    std::vector<MemoryOperation> get_memory_operations() const override;
    double simulate_forward_pass(HardwareSimulator& hw_sim) override;
    
private:
    void initialize_operations();
    
    std::vector<ComputeOperation> compute_ops_;
    std::vector<MemoryOperation> memory_ops_;
};

/**
 * Complete AI model configuration
 */
struct ModelConfig {
    std::string name = "GPT-like-model";
    size_t num_layers = 12;
    size_t hidden_size = 768;
    size_t num_attention_heads = 12;
    size_t intermediate_size = 3072;
    size_t vocab_size = 50000;
    size_t max_sequence_length = 2048;
    size_t batch_size = 1;
    
    // Model architecture variants
    bool use_layer_norm = true;
    bool use_dropout = true;
    std::string activation_function = "gelu";
};

/**
 * Complete AI model simulator
 */
class AIModelSimulator {
public:
    explicit AIModelSimulator(const ModelConfig& config = ModelConfig{});
    
    void set_config(const ModelConfig& config);
    const ModelConfig& get_config() const { return config_; }
    
    // Model construction
    void build_model();
    void add_layer(std::unique_ptr<ModelLayer> layer);
    
    // Simulation
    double simulate_inference(HardwareSimulator& hw_sim, size_t input_tokens = 0);
    double simulate_training_step(HardwareSimulator& hw_sim, size_t input_tokens = 0);
    
    // Analysis
    std::vector<ComputeOperation> get_all_compute_operations() const;
    std::vector<MemoryOperation> get_all_memory_operations() const;
    size_t get_total_parameters() const;
    size_t get_total_flops_per_token() const;
    
    // Model optimization insights
    std::vector<std::string> analyze_computational_bottlenecks() const;
    std::vector<std::string> suggest_model_optimizations() const;
    
    void print_model_summary() const;
    
private:
    ModelConfig config_;
    std::vector<std::unique_ptr<ModelLayer>> layers_;
    bool model_built_ = false;
    
    void create_transformer_layers();
    LayerConfig create_attention_config() const;
    LayerConfig create_feedforward_config() const;
    LayerConfig create_embedding_config() const;
};

} // namespace HighFidelityAI