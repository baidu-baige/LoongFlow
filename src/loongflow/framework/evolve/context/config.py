# -*- coding: utf-8 -*-
"""
EvolveChain configuration.
"""

import os
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class LoggerConfig(BaseModel):
    """Configuration for the logging system."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="The global logging level."
    )
    console_logging: bool = Field(
        default=True, description="Whether to output logs to the console (stdout)."
    )
    file_logging: bool = Field(
        default=True, description="Whether to write logs to a file."
    )
    log_path: Optional[str] = Field(
        default=None,
        description="Directory to store log files. Defaults to '<workspace_path>/logs'.",
    )
    filename: str = Field(default="evolux.log", description="The name of the log file.")
    rotation: str = Field(
        default="H",
        description="Log rotation frequency. 'S' for seconds, 'M' for minutes, 'H' for hours, 'D' for days.",
    )
    backup_count: int = Field(
        default=0,
        description="Number of backup log files to keep. 0 means logs are kept forever.",
    )


class LLMConfig(BaseModel):
    """LLM configuration class."""

    url: str = Field(..., description="The API endpoint URL for the language model.")
    api_key: str = Field(..., description="The API key for authentication.")
    model: str = Field(
        ..., description="The specific model name to use, e.g., 'gpt-4o'."
    )
    model_provider: Optional[str] = Field(
        default=None,
        description="The provider of the model, such as 'openai', 'azure', etc.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Controls randomness. Lower is more deterministic.",
    )
    context_length: int = Field(
        default=65536, gt=0, description="The maximum context length of tokens for the model."
    )
    max_tokens: int = Field(
        default=16384, gt=0, description="The maximum number of tokens to generate."
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Controls nucleus sampling."
    )
    timeout: int = Field(
        default=600,
        description="Timeout in seconds for a single llm generation.",
    )
    completion_token_price: float = Field(
        default=0.0,
        description="Price per token for completion requests.",
    )
    prompt_token_price: float = Field(
        default=0.0,
        description="Price per token for prompt requests.",
    )


class EvaluatorConfig(BaseModel):
    """Evaluator configuration class."""

    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="LLM configuration for the evaluator. Inherits from global if not set.",
    )
    evaluate_code: str = Field(
        ..., description="The Python code string used to evaluate the generated output."
    )
    workspace_path: Optional[str] = Field(
        default=None,
        description="Path to the workspace for storing evaluation artifacts. "
                    "If not set, defaults to a subdirectory within the root workspace.",
    )
    timeout: int = Field(
        default=1800,
        gt=0,
        description="Timeout in seconds for a single evaluation run.",
    )
    evolve_target: Optional[str] = Field(
        default=None,
        description="The specific target or goal for the evolution process, if applicable.",
    )


class DatabaseConfig(BaseModel):
    """Configuration for EvolveDatabase"""

    storage_type: str = Field(default="in_memory", description="Storage type")
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL.",
    )
    num_islands: int = Field(default=3, description="Number of islands")
    population_size: int = Field(default=100, description="Population size")
    elite_archive_size: int = Field(default=50, description="Elite archive size")
    use_sampling_weight: bool = Field(default=True, description="Use sampling weight")
    sampling_weight_power: float = Field(
        default=1.0, description="Sampling weight power"
    )
    exploration_rate: float = Field(default=0.2, description="Exploration rate")
    migration_interval: int = Field(default=10, description="Migration interval")
    migration_rate: float = Field(default=0.2, description="Migration rate")
    boltzmann_temperature: float = Field(
        default=1.0, description="Boltzmann temperature"
    )
    feature_bins: int = Field(default=None, description="Feature bins")
    feature_dimensions: list[str] = Field(
        default=["complexity", "diversity", "score"], description="Feature dimensions"
    )
    feature_scaling_method: str = Field(
        default="minmax", description="Feature scaling method"
    )
    checkpoint_interval: int = Field(
        default=50, description="Checkpoint saving interval"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Path to the directory for database outputs (e.g., checkpoints). "
                    "If not set, defaults to a subdirectory within the root workspace.",
    )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for MemoryFactory"""
        return {
            "storage_type": self.storage_type,
            "redis_url": self.redis_url,
            "num_islands": self.num_islands,
            "population_size": self.population_size,
            "elite_archive_size": self.elite_archive_size,
            "use_sampling_weight": self.use_sampling_weight,
            "sampling_weight_power": self.sampling_weight_power,
            "migration_interval": self.migration_interval,
            "migration_rate": self.migration_rate,
            "boltzmann_temperature": self.boltzmann_temperature,
            "feature_bins": self.feature_bins,
            "feature_dimensions": self.feature_dimensions,
            "feature_scaling_method": self.feature_scaling_method,
            "output_path": self.output_path,
        }


class EvolveConfig(BaseModel):
    """Evolve configuration class."""

    task: str = Field(
        ..., description="The main task or objective for the evolution process."
    )
    initial_code: str = Field(
        default="",
        description="The initial code to start the evolution process.",
    )
    initial_score: float = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The initial score to start the evolution process.",
    )
    initial_evaluation: str = Field(
        default="",
        description="The evaluation result for the evolution process.",
    )
    workspace_path: Optional[str] = Field(
        default=None,
        description="Path to the main workspace for the evolution process. "
                    "If not set, defaults to the root workspace path.",
    )
    database: DatabaseConfig = Field(..., description="Configuration for the database.")
    evaluator: EvaluatorConfig = Field(
        ..., description="Configuration for the evaluation process."
    )
    max_iterations: int = Field(
        default=100, gt=0, description="The maximum number of evolution iterations."
    )
    target_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="The target score to achieve, at which point the evolution will stop.",
    )
    concurrency: int = Field(
        default=5, gt=0, description="The number of concurrent evaluations to run."
    )

    # These fields now act as keys to select the specific configuration from the root level.
    planner_name: str = Field(
        default="evolve_planner",
        description="The name of the planner configuration to use from the root 'planners' map.",
    )
    executor_name: str = Field(
        default="evolve_executor",
        description="The name of the executor configuration to use from the root 'executors' map.",
    )
    summary_name: str = Field(
        default="evolve_summary",
        description="The name of the summarizer configuration to use from the root 'summarizers' map.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for evolve."
    )


class EvolveChainConfig(BaseModel):
    """
    The root configuration model for the entire Evolve Chain process.
    It defines all available components (planners, executors, summarizers)
    and specifies which ones to use for a given evolution run.
    """

    workspace_path: str = Field(
        default="./evolve_run_output",
        description="The root directory for all outputs, logs, and artifacts from the run.",
    )

    logger: Optional[LoggerConfig] = Field(
        default_factory=LoggerConfig,
        description="Logging configuration for the application.",
    )

    llm_config: Optional[LLMConfig] = Field(
        default=None,
        description="Global LLM configuration, used as a fallback for agents and evaluators.",
    )

    # Definitions for all available planners, executors, and summarizers.
    # The inner Dict[str, Any] allows for arbitrary, unstructured configuration for each component.
    planners: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="A dictionary of all available planner configurations, keyed by their unique names.",
    )
    executors: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="A dictionary of all available executor configurations, keyed by their unique names.",
    )
    summarizers: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="A dictionary of all available summarizer configurations, keyed by their unique names.",
    )

    evolve: EvolveConfig = Field(
        ..., description="The main evolution process configuration."
    )

    @model_validator(mode="after")
    def resolve_workspace_paths(self) -> "EvolveChainConfig":
        """
        Automatically sets the workspace paths for subcomponents based on the
        root `workspace_path`, unless they are explicitly overridden.

        This follows a "convention over configuration" principle for ease of use.
        - Evolve Workspace -> <root_workspace>
        - Evaluator Workspace -> <root_workspace>/evaluator
        - Database Output -> <root_workspace>/database
        """
        root_path = self.workspace_path

        # Resolve EvolveConfig's path
        if self.evolve.workspace_path is None:
            # The main evolution workspace is the root itself.
            self.evolve.workspace_path = root_path

        # Resolve EvaluatorConfig's path
        if self.evolve.evaluator.workspace_path is None:
            self.evolve.evaluator.workspace_path = os.path.join(root_path, "evaluator")

        # Resolve DatabaseConfig's path
        if self.evolve.database.output_path is None:
            self.evolve.database.output_path = os.path.join(root_path, "database")

        # Resolve LoggerConfig's path
        if self.logger and self.logger.log_path is None:
            self.logger.log_path = os.path.join(root_path, "logs")

        return self

    @model_validator(mode="after")
    def resolve_evaluator_llm_config(self) -> "EvolveChainConfig":
        """
        Ensures the evaluator has an LLM configuration.
        If the evaluator lacks a specific LLM config, it inherits the global one.
        If no global config is available, it raises an error.
        This validator remains separate to handle the explicitly typed EvaluatorConfig.
        """
        global_llm_config = self.llm_config

        evaluator = self.evolve.evaluator
        if evaluator.llm_config is None:
            if global_llm_config:
                evaluator.llm_config = global_llm_config.model_copy()
            else:
                raise ValueError(
                    "Evaluator is missing an LLM configuration, and no global 'llm_config' is defined."
                )
        return self

    @model_validator(mode="after")
    def validate_and_resolve_agents(self) -> "EvolveChainConfig":
        """
        1. Validates that the selected planner, executor, and summarizer exist.
        2. Injects the global LLM config into the selected agent's config
           if the agent does not have its own 'llm_config' defined.
        """
        # --- 1. Validation ---
        planner_name = self.evolve.planner_name
        if planner_name is not None and planner_name not in self.planners:
            raise ValueError(
                f"Planner '{planner_name}' is not defined in the top-level 'planners' section. "
                f"Available planners: {list(self.planners.keys())}"
            )

        executor_name = self.evolve.executor_name
        if executor_name is not None and executor_name not in self.executors:
            raise ValueError(
                f"Executor '{executor_name}' is not defined in the top-level 'executors' section. "
                f"Available executors: {list(self.executors.keys())}"
            )

        summary_name = self.evolve.summary_name
        if summary_name is not None and summary_name not in self.summarizers:
            raise ValueError(
                f"Summarizer '{summary_name}' is not defined in the top-level 'summarizers' section. "
                f"Available summarizers: {list(self.summarizers.keys())}"
            )

        # --- 2. LLM Config Injection ---
        # Only proceed if a global LLM config is available.
        if self.llm_config:
            # Helper function to inject config to avoid repetition
            def _inject_llm_if_missing(agent_config: Dict[str, Any]):
                if "llm_config" not in agent_config:
                    # Use model_copy() to avoid multiple components modifying the same object instance.
                    agent_config["llm_config"] = self.llm_config.model_copy()

            # Inject into the selected planner, executor, and summarizer
            _inject_llm_if_missing(self.planners[planner_name or "evolve_planner"])
            _inject_llm_if_missing(self.executors[executor_name or "evolve_executor"])
            _inject_llm_if_missing(self.summarizers[summary_name or "evolve_summary"])

        return self


def load_config(config_path: str) -> EvolveChainConfig:
    """
    Loads, parses, and validates the YAML configuration file.

    Args:
        config_path: The path to the config.yaml file.

    Returns:
        A fully validated and resolved EvolveChainConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the configuration is invalid.
        yaml.YAMLError: If the YAML syntax is incorrect.
    """

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise

    try:
        config = EvolveChainConfig.model_validate(config_data)
        return config
    except ValidationError as e:
        print(f"Configuration validation error:\n{e}")
        raise
