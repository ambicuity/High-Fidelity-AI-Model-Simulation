#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

namespace HighFidelityAI {

// Forward declarations
struct HardwareConfig;
struct ModelConfig;
struct PerformanceMetrics;
class HardwareSimulator;
class AIModelSimulator;

/**
 * Optimization objective types
 */
enum class OptimizationObjective {
    MINIMIZE_LATENCY,
    MINIMIZE_ENERGY,
    MAXIMIZE_THROUGHPUT,
    MINIMIZE_COST,
    BALANCED_PERFORMANCE
};

/**
 * Hardware optimization parameter
 */
struct OptimizationParameter {
    std::string name;
    double min_value;
    double max_value;
    double current_value;
    double step_size;
    
    OptimizationParameter(const std::string& n, double min_val, double max_val, double current)
        : name(n), min_value(min_val), max_value(max_val), current_value(current), 
          step_size((max_val - min_val) / 20.0) {}
};

/**
 * Optimization result
 */
struct OptimizationResult {
    HardwareConfig optimal_hardware_config;
    ModelConfig optimal_model_config;
    PerformanceMetrics achieved_metrics;
    double objective_score = 0.0;
    size_t iterations_performed = 0;
    std::vector<std::string> optimization_steps;
    
    void print_summary() const;
};

/**
 * Base class for optimization algorithms
 */
class OptimizationAlgorithm {
public:
    explicit OptimizationAlgorithm(OptimizationObjective objective) : objective_(objective) {}
    virtual ~OptimizationAlgorithm() = default;
    
    virtual OptimizationResult optimize(
        HardwareSimulator& hw_sim,
        AIModelSimulator& model_sim,
        const std::vector<OptimizationParameter>& parameters,
        size_t max_iterations = 100) = 0;
    
    void set_objective(OptimizationObjective objective) { objective_ = objective; }
    OptimizationObjective get_objective() const { return objective_; }
    
protected:
    OptimizationObjective objective_;
    
    double evaluate_objective(const PerformanceMetrics& metrics) const;
    void log_optimization_step(const std::string& step, std::vector<std::string>& log) const;
};

/**
 * Grid search optimization algorithm
 */
class GridSearchOptimizer : public OptimizationAlgorithm {
public:
    explicit GridSearchOptimizer(OptimizationObjective objective) 
        : OptimizationAlgorithm(objective) {}
    
    OptimizationResult optimize(
        HardwareSimulator& hw_sim,
        AIModelSimulator& model_sim,
        const std::vector<OptimizationParameter>& parameters,
        size_t max_iterations = 100) override;
    
    void set_grid_resolution(size_t resolution) { grid_resolution_ = resolution; }
    
private:
    size_t grid_resolution_ = 5;
    
    std::vector<std::vector<double>> generate_grid_points(
        const std::vector<OptimizationParameter>& parameters) const;
};

/**
 * Genetic algorithm optimizer
 */
class GeneticAlgorithmOptimizer : public OptimizationAlgorithm {
public:
    explicit GeneticAlgorithmOptimizer(OptimizationObjective objective)
        : OptimizationAlgorithm(objective) {}
    
    OptimizationResult optimize(
        HardwareSimulator& hw_sim,
        AIModelSimulator& model_sim,
        const std::vector<OptimizationParameter>& parameters,
        size_t max_iterations = 100) override;
    
    void set_population_size(size_t size) { population_size_ = size; }
    void set_mutation_rate(double rate) { mutation_rate_ = rate; }
    void set_crossover_rate(double rate) { crossover_rate_ = rate; }
    
private:
    size_t population_size_ = 20;
    double mutation_rate_ = 0.1;
    double crossover_rate_ = 0.8;
    
    struct Individual {
        std::vector<double> genes;
        double fitness = 0.0;
    };
    
    std::vector<Individual> initialize_population(
        const std::vector<OptimizationParameter>& parameters) const;
    void evaluate_population(std::vector<Individual>& population,
                           HardwareSimulator& hw_sim,
                           AIModelSimulator& model_sim,
                           const std::vector<OptimizationParameter>& parameters);
    std::vector<Individual> select_parents(const std::vector<Individual>& population) const;
    Individual crossover(const Individual& parent1, const Individual& parent2) const;
    void mutate(Individual& individual, const std::vector<OptimizationParameter>& parameters) const;
};

/**
 * Bayesian optimization algorithm
 */
class BayesianOptimizer : public OptimizationAlgorithm {
public:
    explicit BayesianOptimizer(OptimizationObjective objective)
        : OptimizationAlgorithm(objective) {}
    
    OptimizationResult optimize(
        HardwareSimulator& hw_sim,
        AIModelSimulator& model_sim,
        const std::vector<OptimizationParameter>& parameters,
        size_t max_iterations = 100) override;
    
private:
    // Simplified Gaussian Process implementation
    struct GaussianProcess {
        std::vector<std::vector<double>> X;  // Input points
        std::vector<double> y;               // Output values
        double length_scale = 1.0;
        double noise_level = 0.01;
        
        void fit(const std::vector<std::vector<double>>& inputs,
                const std::vector<double>& outputs);
        std::pair<double, double> predict(const std::vector<double>& x) const;
    };
    
    GaussianProcess gp_model_;
    
    std::vector<double> acquisition_function(const std::vector<std::vector<double>>& candidates,
                                           const GaussianProcess& gp) const;
    std::vector<double> generate_next_point(const std::vector<OptimizationParameter>& parameters,
                                          const GaussianProcess& gp) const;
};

/**
 * Multi-objective optimization result
 */
struct MultiObjectiveResult {
    std::vector<OptimizationResult> pareto_front;
    std::vector<std::string> trade_off_analysis;
    
    void print_pareto_summary() const;
};

/**
 * Multi-objective optimizer using NSGA-II algorithm
 */
class MultiObjectiveOptimizer {
public:
    MultiObjectiveOptimizer(const std::vector<OptimizationObjective>& objectives)
        : objectives_(objectives) {}
    
    MultiObjectiveResult optimize(
        HardwareSimulator& hw_sim,
        AIModelSimulator& model_sim,
        const std::vector<OptimizationParameter>& parameters,
        size_t max_iterations = 100);
    
    void set_population_size(size_t size) { population_size_ = size; }
    
private:
    std::vector<OptimizationObjective> objectives_;
    size_t population_size_ = 50;
    
    struct MultiObjectiveIndividual {
        std::vector<double> genes;
        std::vector<double> objectives;
        size_t domination_count = 0;
        std::vector<size_t> dominated_solutions;
        size_t rank = 0;
        double crowding_distance = 0.0;
    };
    
    std::vector<std::vector<MultiObjectiveIndividual>> fast_non_dominated_sort(
        std::vector<MultiObjectiveIndividual>& population) const;
    void calculate_crowding_distance(std::vector<MultiObjectiveIndividual>& front) const;
    bool dominates(const MultiObjectiveIndividual& a, const MultiObjectiveIndividual& b) const;
};

/**
 * Hardware-software co-design optimizer
 */
class CoDesignOptimizer {
public:
    CoDesignOptimizer() = default;
    
    // Joint optimization of hardware and model parameters
    OptimizationResult optimize_joint_design(
        const std::vector<OptimizationParameter>& hardware_params,
        const std::vector<OptimizationParameter>& model_params,
        OptimizationObjective primary_objective = OptimizationObjective::BALANCED_PERFORMANCE,
        size_t max_iterations = 200);
    
    // Design space exploration
    MultiObjectiveResult explore_design_space(
        const std::vector<OptimizationParameter>& hardware_params,
        const std::vector<OptimizationParameter>& model_params,
        const std::vector<OptimizationObjective>& objectives,
        size_t num_samples = 1000);
    
    // Trade-off analysis
    std::vector<std::string> analyze_hardware_software_tradeoffs(
        const MultiObjectiveResult& results) const;
    
    void print_codesign_recommendations(const OptimizationResult& result) const;
    
private:
    std::unique_ptr<OptimizationAlgorithm> create_optimizer(OptimizationObjective objective) const;
    
    void validate_parameter_combinations(
        const std::vector<OptimizationParameter>& hw_params,
        const std::vector<OptimizationParameter>& model_params) const;
};

} // namespace HighFidelityAI