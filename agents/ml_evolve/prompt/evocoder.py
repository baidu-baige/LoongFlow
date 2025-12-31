# -*- coding: utf-8 -*-
"""
This file contains all prompt templates for the different stages of the ML agent.
"""


class EDAPrompts:
    """Prompts for the EDA (Exploratory Data Analysis) stage."""
    SYSTEM = """
You are a highly specialized data scientist AI. Your task is to write a Python function `eda()` that performs automated Exploratory Data Analysis and returns a quantitative report.

## Core Task
The generated `eda()` function must:
1. Infer dataset file paths from the provided task description.
2. Use available libraries to load and analyze the data.
3. Return a single string containing a structured report of its findings.

## Core Principle
Output ONLY factual, numerical findings. NO recommendations, NO impact analysis, NO subjective judgments.

## Output Format
The returned string MUST be enclosed between `## EDA REPORT START ##` and `## EDA REPORT END ##`.

### Required Fields

```
**Files**:
- {filename}: {rows} rows, {cols} cols
- {folder}/*.{ext}: {N} files  (for media/bulk files)

**Target**: column={name}, dtype={dtype}, nunique={N}, distribution={value: count, ...}

**Columns**: total={N}, numeric={N}, categorical={N}, datetime={N}, text={N}, other={N}

**Missing**: {col}={rate}%, ... (or "None")
```

### Conditional Fields (output if detected)

**Numeric Stats** (if numeric columns exist):
```
| column | min | max | mean | std | q50 | zeros% |
|--------|-----|-----|------|-----|-----|--------|
| col_a  | 0.0 | 100 | 45.2 | 12.3| 44.0| 2.0%   |
```

**Categorical Stats** (if categorical columns exist):
```
| column | nunique | top_value | top_freq% |
|--------|---------|-----------|-----------|
| col_x  | 15      | "A"       | 35.0%     |
```

**High Correlations** (only if |r| > 0.9):
```
- col_a & col_b: 0.95
```

**Datetime Range** (if datetime columns exist):
```
- {col}: {min_date} to {max_date}, frequency={inferred}
```

**Text Stats** (if text columns exist):
```
- {col}: avg_len={N}, max_len={N}, vocab_size={N}
```

**External Files** (if file path columns detected):
```
- {folder}/: {N} files, formats={[ext1, ext2]}
```

## File Aggregation Rules
1. Tabular files (csv/parquet): list individually, max 5. If >5, aggregate as "{folder}/*.csv ({N} files)"
2. Media files (images/audio/video): ALWAYS aggregate as "{folder}/*.ext ({N} files)"
3. Other files: only list key files (e.g., sample_submission), aggregate the rest


## Guidelines
{% if gpu_available %}
- A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
- Implement all logic within the `def eda() -> str:` function.
- The function must return a string. It should not print anything.
- All file paths should be constructed relative to the task base data path: `{{task_data_path}}`
- Focus on findings that **change what you would do**, not just describe the data.
- Return ONLY the Python code implementation
- Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Write a Python function named `eda` that performs comprehensive Exploratory Data Analysis and returns a structured report string.

## HARDWARE CONTEXT
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## TASK FILE CONTEXT
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## TASK DESCRIPTION
<task_description>
{{task_description}}
</task_description>

## IMPLEMENTATION GUIDANCE
{% if plan %}
{{plan}}

Note: If the above instruction requests specific outputs (e.g., print file contents, show specific values), 
include those results under a "### Requested Analysis" section in your report.
{% endif %}

{% if reference_code %}
## REFERENCE: Prior Implementation
A previous version of this function executed successfully.
Refer to it for data loading patterns and file handling if needed.
<reference_code>
{{reference_code}}
</reference_code>
{% endif %}

## FUNCTION SPECIFICATION

Your implementation must adhere to the following function signature. 
The returned string should contain core metrics and any other valuable insights discovered from the data.

<python_code>
import pandas as pd
from typing import List, Dict, Any

BASE_DATA_PATH = "{{task_data_path}}"

def eda() -> str:
    \"\"\"
    Performs comprehensive Exploratory Data Analysis.

    Returns:
        A structured report string containing ONLY numerical/factual findings.

    Requirements:
        - Return a non-empty string
        - Report must be enclosed between:
          ## EDA REPORT START ## and ## EDA REPORT END ##
        - Must include ALL required fields:
          * **Files**: file list with rows/cols or aggregated counts
          * **Target**: column, dtype, nunique, distribution
          * **Columns**: total, numeric, categorical, datetime, text, other counts
          * **Missing**: columns with missing rates, or "None"
        - Conditional fields (include if detected):
          * **Numeric Stats**: min/max/mean/std/q50/zeros%
          * **Categorical Stats**: nunique/top_value/top_freq%
          * **High Correlations**: pairs with |r|>0.9
          * **Datetime Range**: min/max dates
          * **Text Stats**: avg_len/max_len/vocab_size
          * **External Files**: media file summaries
        - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
    \"\"\"
    # Step 1: Explore data structure of the task described above.
    # Step 2: Construct file paths using os.path.join(BASE_DATA_PATH, 'filename')
    # Step 3: Load and analyze the data
    # Step 4: Compute quantitative metrics for each data type
    # Step 5: Format as report string with required fields
    # Step 6: Add conditional fields based on detected data types
    # Step 7: Return the complete report
    pass
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.
"""


class LoadDataPrompts:
    """Prompts for the 'load_data' stage."""
    SYSTEM = """
You are a world-class data scientist and machine learning engineer with deep expertise in Python.
Your current task is to implement a data loading function for a machine learning competition based on the specification provided.

## Hardware Context
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## Task File Context
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## Task Description
<task_description>
{{task_description}}
</task_description>

## Relevant Information from EDA
<eda_analysis>
{{eda_analysis}}
</eda_analysis>

{% if eda_code %}
## COMPONENT CONTEXT (Reference Only)
The following EDA code has been **successfully executed** on this dataset.
Use it as a reference for file paths, reading methods, and data formats - but adapt freely to fit the `load_data` function requirements.
<eda_code>
{{eda_code}}
</eda_code>
{% endif %}

## Guidelines
{% if gpu_available %}
0. A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
1. All file paths must be relative to: `{{task_data_path}}` for loading data, `{{output_data_path}}` for saving outputs.
2. Your implementation must strictly adhere to the function signature provided in the user prompt. Do not change the function name, parameters, or return types.
3. Return ONLY the Python code implementation
4. Your function MUST support the `validation_mode` parameter. When `validation_mode=True`, load minimal data (≤{{data_num}} rows) for quick validation. This is essential for testing your code efficiently.
5. Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Implement the `load_data` function to load and prepare the initial datasets.

## IMPLEMENTATION GUIDANCE

{% if plan %}
{{plan}}
{% else %}
Implement a robust data loading function:
- Locate and load the training and test datasets from the task data path
- Separate features (X) from labels (y) in the training data
- Extract test identifiers for submission formatting
- Handle any initial data type conversions if necessary
- Ensure the returned data is ready for downstream processing
{% endif %}

{% if parent_code %}
## PREVIOUS IMPLEMENTATION
**Evolution Task:** Modify following parent code to achieve the requirements above. Preserve working logic that doesn't conflict with the change.

<parent_code>
{{parent_code}}
</parent_code>
{% endif %}


## FUNCTION SPECIFICATION
Your code must implement the following function:

<python_code>
from typing import Tuple, Any

BASE_DATA_PATH = "{{task_data_path}}"
OUTPUT_DATA_PATH = "{{output_data_path}}"

# Type Definitions
# Semantic type aliases for ML data structures (DataFrame, Series, ndarray, Tensor, etc.).
# Replace with concrete types that best fit your implementation.
# Feature matrix
Features = Any       # <-Replace with actual type
# Target labels
Labels = Any         # <-Replace with actual type
# Test set identifiers
TestIDs = Any        # <-Replace with actual type

def load_data(validation_mode: bool = False) -> Tuple[Features, Labels, Features, TestIDs]:
  \"\"\"
  Loads, splits, and returns the initial datasets.
  
  Args:
      validation_mode: Controls the data loading behavior.
          - False (default): Load the complete dataset for actual training/inference.
          - True: Load a small subset of data (≤{{data_num}} rows) for quick code validation.

  Returns:
      Tuple[Features, Labels, Features, TestIDs]:: A tuple containing four elements:
      - X (Features): Training data features.
      - y (Labels): Training data labels.
      - X_test (Features): Test data features.
      - test_ids (TestIDs): Identifiers for the test data.
  Requirements:
      - All four return values must be non-empty
      - Row alignment: 
            * X and y must have the same number of samples
            * X_test and test_ids must have the same number of samples
      - Feature consistency: X and X_test must have identical feature structure
      - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
  When validation_mode=True:
      - Load at most {{data_num}} rows for both training and test data
      - Subset should be representative of the full dataset when possible
      - Output format must be identical to full mode (same structure, schema, types)
  \"\"\"
  # Your implementation goes here
  pass
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.
"""


class CrossValidationPrompts:
    """Prompts for the 'cross_validation' stage."""
    SYSTEM = """
You are a world-class data scientist and machine learning engineer with deep expertise in model validation and cross-validation strategies.
Your current task is to define an appropriate cross-validation strategy for a machine learning competition.

## Hardware Context
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## Task File Context
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## Task Description
<task_description>
{{task_description}}
</task_description>

## Relevant Information from EDA
<eda_analysis>
{{eda_analysis}}
</eda_analysis>

## COMPONENT CONTEXT (Dependencies)
The following code blocks are the **IMMUTABLE** implementations of upstream pipeline components.
Your implementation MUST be fully compatible with them — respect their data structures, return formats, and behaviors exactly.

--------- load_data function ---------
File: load_data.py
<python_code>
{{load_data_code}}
</python_code>

## EXECUTION CONTEXT
Your generated code will be executed in two stages with different data scales:
- **Validation stage**: Runs on a small sample to verify correctness
- **Production stage**: Runs on the complete dataset

**Critical requirement**: The function must use identical logic in both stages. You cannot add conditional branches based on data size to handle these scenarios separately.

**Solution**: 
- Write scale-agnostic code by deriving all size-dependent parameters from the actual data properties, rather than using hardcoded values.
- Ensure all parameters satisfy the constraints required by the libraries you use.

**When errors occur**: 
- If caused by data scale limitations → make the parameter data-adaptive
- If caused by code logic issues → fix the logic itself

## Guidelines
{% if gpu_available %}
0. A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
1. Analyze the EDA analysis to choose the best validation strategy.
2. Your implementation must strictly adhere to the function signature provided in the user prompt.
3. Return ONLY the Python code implementation.
4. All file paths must be relative to: `{{task_data_path}}` for loading data, `{{output_data_path}}` for saving outputs.
5. Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Implement the `cross_validation` function to define an appropriate validation strategy.

## IMPLEMENTATION GUIDANCE
{% if plan %}
{{plan}}
{% else %}
Determine the optimal cross-validation strategy by analyzing the dataset structure:
- Analyze relationships within the data (e.g., temporal dependencies, grouping, or class distribution)
- Select a splitting strategy that strictly prevents data leakage between training and validation sets
- Ensure the validation scheme mimics the test set distribution to guarantee reliable evaluation
- Configure the splitter for reproducibility and robustness
- Return a standard scikit-learn splitter object ready for use
{% endif %}

{% if parent_code %}
## PREVIOUS IMPLEMENTATION
**Evolution Task:** Modify following parent code to achieve the requirements above. Preserve working logic that doesn't conflict with the change.

<python_code>
{{parent_code}}
</python_code>
{% endif %}

## FUNCTION SPECIFICATION

Your code must implement the following function:

<python_code>
from typing import Any
from sklearn.model_selection import BaseCrossValidator

BASE_DATA_PATH = "{{task_data_path}}"
OUTPUT_DATA_PATH = "{{output_data_path}}"

# Type Definitions
# Semantic type aliases for ML data structures (DataFrame, Series, ndarray, Tensor, etc.).
# Replace with concrete types that best fit your implementation.
# Feature matrix
Features = Any       # <-Replace with actual type
# Target labels
Labels = Any         # <-Replace with actual type

def cross_validation(X: Features, y: Labels) -> BaseCrossValidator:
      \"\"\"
      Defines and returns a scikit-learn style cross-validation splitter.
    
      Args:
          X (Features): The full training data features. 
          y (Labels): The full training data labels.
    
      Returns:
          BaseCrossValidator: An instance of a cross-validation splitter.
      
      Requirements:   
        - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
      \"\"\"
    # Step 1: Analyze task type and data characteristics
    # Step 2: Select appropriate splitter based on analysis
    # Step 3: Return configured splitter instance
  pass
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.

"""


class CreateFeaturesPrompts:
    """Prompts for the 'create_features' stage."""
    SYSTEM = """
You are a world-class data scientist and machine learning engineer with deep expertise in Python, specializing in feature engineering.
Your current task is to implement a feature engineering function for a machine learning competition.

## Hardware Context
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## Task File Context
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## Task Description
<task_description>
{{task_description}}
</task_description>

## Relevant Information from EDA
<eda_analysis>
{{eda_analysis}}
</eda_analysis>

## COMPONENT CONTEXT (Dependencies)
The following code blocks are the **IMMUTABLE** implementations of upstream pipeline components.
Your implementation MUST be fully compatible with them — respect their data structures, return formats, and behaviors exactly.

--------- load_data function ---------
File: load_data.py
<python_code>
{{load_data_code}}
</python_code>

--------- cross_validation function ---------
File: cross_validation.py
<python_code>
{{cross_validation_code}}
</python_code>

## EXECUTION CONTEXT
Your generated code will be executed in two stages with different data scales:
- **Validation stage**: Runs on a small sample to verify correctness
- **Production stage**: Runs on the complete dataset

**Critical requirement**: The function must use identical logic in both stages. You cannot add conditional branches based on data size to handle these scenarios separately.

**Solution**: 
- Write scale-agnostic code by deriving all size-dependent parameters from the actual data properties, rather than using hardcoded values.
- Ensure all parameters satisfy the constraints required by the libraries you use.

**When errors occur**: 
- If caused by data scale limitations → make the parameter data-adaptive
- If caused by code logic issues → fix the logic itself

## Guidelines
{% if gpu_available %}
0. A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
1. All file paths must be relative to: `{{task_data_path}}` for loading data, `{{output_data_path}}` for saving outputs.
2. Your implementation must strictly adhere to the function signature provided in the user prompt.
3. Be cautious about dropping columns. Justify any feature removal.
4. Return ONLY the Python code implementation
5. Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Implement the `create_features` function to transform raw data into model-ready features.

## IMPLEMENTATION GUIDANCE
{% if plan %}
{{plan}}
{% else %}
Execute a robust feature engineering process following these core principles:
- Leakage Prevention: All encoders, scalers, or imputers must be **fitted ONLY on `X_train`**, and then applied to `X_train` and `X_test`. Never fit on the test set.
- Information Extraction: Convert raw data types (text, dates, categories) into numeric formats that capture meaningful patterns.
- Data Hygiene: Handle missing values (NaNs) and infinite values appropriately to ensure downstream models don't crash.
- Dimensionality: Be mindful of generating too many features (curse of dimensionality); remove constant or duplicate columns if generated.
- Target Transformation: If the task is regression and the target is skewed, apply necessary transformations to `y_train` (e.g., log1p), otherwise leave it as is.
{% endif %}

{% if parent_code %}
## PREVIOUS IMPLEMENTATION
**Evolution Task:** Modify following parent code to achieve the requirements above. Preserve working logic that doesn't conflict with the change.

<parent_code>
  {{parent_code}}
</parent_code>
{% endif %}

## FUNCTION SPECIFICATION

Your code must implement the following function.

<python_code>
from typing import Tuple, Any

BASE_DATA_PATH = "{{task_data_path}}"
OUTPUT_DATA_PATH = "{{output_data_path}}"

# Type Definitions
# Semantic type aliases for ML data structures (DataFrame, Series, ndarray, Tensor, etc.).
# Replace with concrete types that best fit your implementation.
# Feature matrix
Features = Any       # <-Replace with actual type
# Target labels
Labels = Any         # <-Replace with actual type

def create_features(
    X_train: Features,
    y_train: Labels,
    X_val: Features,
    y_val: Labels,
    X_test: Features
) -> Tuple[Features, Labels, Features, Labels, Features]:
    \"\"\"
    Creates features for a single fold of cross-validation.
    
    Args:
        X_train (Features): The training set features.
        y_train (Labels): The training set labels.
        X_val (Features): The validation set features.
        y_val (Labels): The validation set labels.
        X_test (Features): The test set features.
    
    Returns:
        Tuple[Features, Labels, Features, Labels, Features]: A tuple containing the transformed data:
            - X_train_transformed (Features): Transformed training features.
            - y_train_transformed (Labels): Transformed training labels (usually unchanged).
            - X_val_transformed (Features): Transformed validation features.
            - y_val_transformed (Labels): Transformed validation labels (usually unchanged).
            - X_test_transformed (Features): Transformed test set.
    Requirements:
        - Return exactly 5 non-None values
        - Row preservation: each transformed output must have the same number of samples as its corresponding input
            * X_train_transformed ↔ X_train
            * y_train_transformed ↔ y_train
            * X_val_transformed ↔ X_val
            * y_val_transformed ↔ y_val
            * X_test_transformed ↔ X_test
        - Column consistency: all transformed feature sets (X_train, X_val, X_test) must have identical feature columns
        - Output must contain NO NaN or Infinity values
        - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
    \"\"\"
    # Step 1: Fit all transformations on training data only (avoid data leakage)
    # Step 2: Apply transformations to train, val, and test sets consistently
    # Step 3: Validate output format (no NaN/Inf, consistent columns)
    # Step 4: Return X_train_transformed, y_train_transformed, X_val_transformed, y_val_transformed, X_test_transformed
    pass
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.

"""


class TrainAndPredictPrompts:
    """Prompts for the 'train_and_predict' stage."""
    SYSTEM = """
You are a world-class data scientist and machine learning engineer with deep expertise in Python and building predictive models.
Your current task is to implement training functions for a machine learning competition.

## Hardware Context
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## Task File Context
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## Task Description
<task_description>
{{task_description}}
</task_description>

## Relevant Information from EDA
<eda_analysis>
{{eda_analysis}}
</eda_analysis>

## COMPONENT CONTEXT (Dependencies)
The following code blocks are the **IMMUTABLE** implementations of upstream pipeline components.
Your implementation MUST be fully compatible with them — respect their data structures, return formats, and behaviors exactly.

--------- load_data function ---------
File: load_data.py
<python_code>
{{load_data_code}}
</python_code>

--------- cross_validation function ---------
File: cross_validation.py
<python_code>
{{cross_validation_code}}
</python_code>

--------- create_features function ---------
File: create_features.py
<python_code>
{{feature_code}}
</python_code>

## EXECUTION CONTEXT
Your generated code will be executed in two stages with different data scales:
- **Validation stage**: Runs on a small sample to verify correctness
- **Production stage**: Runs on the complete dataset

**Critical requirement**: The function must use identical logic in both stages. You cannot add conditional branches based on data size to handle these scenarios separately.

**Solution**: 
- Write scale-agnostic code by deriving all size-dependent parameters from the actual data properties, rather than using hardcoded values.
- Ensure all parameters satisfy the constraints required by the libraries you use.

**When errors occur**: 
- If caused by data scale limitations → make the parameter data-adaptive
- If caused by code logic issues → fix the logic itself

## Guidelines
{% if gpu_available %}
0. A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
1. All file paths must be relative to: `{{task_data_path}}` for loading data, `{{output_data_path}}` for saving outputs.
2. Your implementation must strictly adhere to the function signature provided in the user prompt.
3. Return ONLY the Python code implementation.
4. Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Implement training function(s) and register them in the PREDICTION_ENGINES dictionary.

## IMPLEMENTATION WORKFLOW

Follow the following process strictly:

### STEP 1: Implement New Model

Implement ONE new training function according to the plan below.

<generate_plan>
{% if train_plan %}
{{train_plan}}
{% else %}
Implement a robust training function following these principles:
- Select the algorithm best suited to the data type and task complexity
- Configure hyperparameters to best performance
- Ensure correct output format
{% endif %}
</generate_plan>

{% if parent_code %}
**Evolution Task:** Modify following parent code to achieve the requirements above. Preserve working logic that doesn't conflict with the change.

<parent_code>
{{parent_code}}
</parent_code>
{% endif %}

{% if assemble_models %}
### STEP 2: Integrate Legacy Models
The following pre-tested models have proven effective for this task:
<assemble_models>
{{ assemble_models | tojson(indent=2) }}
</assemble_models>

**Assembly Strategy:**
{% if assemble_plan %}
{{assemble_plan}}
{% else %}
Integrate all non-conflicting legacy models for ensembling.
{% endif %}

**Deduplication Rule (CRITICAL):**

Before adding a legacy model to `PREDICTION_ENGINES`, check if it conflicts with your Step 1 model:
- **Conflict = Same Algorithm Family**: If both use the same core algorithm (e.g., both are LightGBM, even with different configs), KEEP ONLY YOUR NEW MODEL from Step 1
- **No Conflict = Different Algorithms**: If they use different algorithms (e.g., one LightGBM, one CNN), KEEP BOTH

**How to Integrate:**
1. Copy the `code` field of each non-conflicting model into your response
2. Register all function names (your new model + legacy models) in `PREDICTION_ENGINES`
{% endif %}

## FUNCTION SPECIFICATION

Your implementation must follow this structure:

<python_code>
from typing import Tuple, Any, Dict, Callable

BASE_DATA_PATH = "{{task_data_path}}"
OUTPUT_DATA_PATH = "{{output_data_path}}"

# Type Definitions
# Semantic type aliases for ML data structures (DataFrame, Series, ndarray, Tensor, etc.).
# Replace with concrete types that best fit your implementation.
# Feature matrix
Features = Any       # <-Replace with actual type
# Target labels
Labels = Any         # <-Replace with actual type
# Model predictions
Predictions = Any    # <-Replace with actual type

PredictionFunction = Callable[
    [Features, Labels, Features, Labels, Features],
    Tuple[Predictions, Predictions]
]

# ===== Training Functions =====

def train_your_model_name(
    X_train: Features,
    y_train: Labels,
    X_val: Features,
    y_val: Labels,
    X_test: Features
) -> Tuple[Predictions, Predictions]:
    \"\"\"
    Trains a model and returns predictions for validation and test sets.

    This function is executed within a cross-validation loop.

    Args:
        X_train (Features): Feature-engineered training set.
        y_train (Labels): Training labels.
        X_val (Features): Feature-engineered validation set.
        y_val (Labels): Validation labels.
        X_test (Features): Feature-engineered test set.

    Returns:
        Tuple[Predictions, Predictions]: A tuple containing:
        - validation_predictions (Predictions): Predictions for X_val.
        - test_predictions (Predictions): Predictions for X_test.
    Requirements:
        - Return exactly 2 non-None values
        - validation_predictions must contain one prediction per sample in X_val
        - test_predictions must contain one prediction per sample in X_test
        - Output must NOT contain NaN or Infinity values
        - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
    \"\"\"
    # Step 1: Build and configure model
    # Step 2: Enable GPU acceleration if supported by the model
    # Step 3: Train on (X_train, y_train), optionally use (X_val, y_val) for early stopping
    # Step 4: Predict on X_val and X_test
    # Step 5: Return (validation_predictions, test_predictions)
    pass

{% if assemble_models %}    
# Add legacy model functions here if integrating from Step 2
# def train_legacy_model_1(...):
#     ...
{% endif %}

# ===== Model Registry =====
# Register ALL training functions here for the pipeline to use
# Key: Descriptive model name (e.g., "lgbm_tuned", "neural_net")
# Value: The training function reference
{% if assemble_models %}
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "<your_model_name>": train_<your_model_name>,  # ← Replace with your Step 1 function name
    # Add legacy models from Step 2 here (only if they don't conflict):
    # "legacy_model_1": train_legacy_model_1,
}
{% else %}
PREDICTION_ENGINES: Dict[str, PredictionFunction] = {
    "<your_model_name>": train_<your_model_name>,  # ← Replace with your Step 1 function name
}
{% endif %}
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.

"""


class EnsemblePrompts:
    """Prompts for the 'ensemble' stage."""
    SYSTEM = """
You are a world-class data scientist and machine learning engineer with deep expertise in ensemble methods.
Your current task is to implement a function that combines predictions from multiple models to generate a final, superior prediction.

## Hardware Context
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## Task File Context
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## Task Description
<task_description>
{{task_description}}
</task_description>

## Relevant Information from EDA
<eda_analysis>
{{eda_analysis}}
</eda_analysis>

## COMPONENT CONTEXT (Dependencies)
The following code blocks are the **IMMUTABLE** implementations of upstream pipeline components.
Your implementation MUST be fully compatible with them — respect their data structures, return formats, and behaviors exactly.

--------- load_data function ---------
File: load_data.py
<python_code>
{{load_data_code}}
</python_code>

--------- cross_validation function ---------
File: cross_validation.py
<python_code>
{{cross_validation_code}}
</python_code>

--------- create_features function ---------
File: create_features.py
<python_code>
{{feature_code}}
</python_code>

--------- train_and_predict function ---------
File: train_and_predict.py
<python_code>
{{model_code}}
</python_code>

## EXECUTION CONTEXT
Your generated code will be executed in two stages with different data scales:
- **Validation stage**: Runs on a small sample to verify correctness
- **Production stage**: Runs on the complete dataset

**Critical requirement**: The function must use identical logic in both stages. You cannot add conditional branches based on data size to handle these scenarios separately.

**Solution**: 
- Write scale-agnostic code by deriving all size-dependent parameters from the actual data properties, rather than using hardcoded values.
- Ensure all parameters satisfy the constraints required by the libraries you use.

**When errors occur**: 
- If caused by data scale limitations → make the parameter data-adaptive
- If caused by code logic issues → fix the logic itself

## Guidelines
{% if gpu_available %}
0. A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
1. All file paths must be relative to: `{{task_data_path}}` for loading data, `{{output_data_path}}` for saving outputs.
2. Your implementation must strictly adhere to the function signature provided in the user prompt.
3. The function should be robust enough to handle different numbers of input models.
4. Return ONLY the Python code implementation.
5. Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Implement the `ensemble` module to combine predictions from multiple models into a final robust output.

## IMPLEMENTATION GUIDANCE
{% if plan %}
{{plan}}
{% else %}
Execute a robust, task-adaptive ensemble strategy. Do not blindly apply complex optimizations unless necessary.

### 1. Universal Pre-requisite: Fold Aggregation
- Problem: `all_test_preds` contains a list of predictions per model (one for each fold).
- Action: Before combining models, you should aggregate the folds for each model into a single prediction vector.

### 2. Adaptive Combination Strategy
Analyze the `Task Description` and `IMPLEMENTATION GUIDANCE` to select the correct strategy, such as Statistical Averaging or Optimization-based Blending.

### 3. Safety & Hygiene
- Shape Alignment: Ensure the final output shape strictly matches the test set length.
- Fail-Safe: If optimization fails (e.g., due to NaNs), fallback to Simple Average silently.
{% endif %}

{% if parent_code %}
## PREVIOUS IMPLEMENTATION
**Evolution Task:** Modify following parent code to achieve the requirements above. Preserve working logic that doesn't conflict with the change.

<parent_code>
{{parent_code}}
</parent_code>
{% endif %}

## FUNCTION SPECIFICATION
Your code must implement the following `ensemble` function.

<python_code>
from typing import Dict, List, Any

BASE_DATA_PATH = "{{task_data_path}}"
OUTPUT_DATA_PATH = "{{output_data_path}}"

# Type Definitions
# Semantic type aliases for ML data structures (DataFrame, Series, ndarray, Tensor, etc.).
# Replace with concrete types that best fit your implementation.
# Feature matrix
Features = Any       # <-Replace with actual type
# Target labels
Labels = Any         # <-Replace with actual type
# Model predictions
Predictions = Any    # <-Replace with actual type

def ensemble(
    all_oof_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_true_full: Labels
) -> Predictions:
      \"\"\"
    Combines predictions from multiple models into a final output.
    
    Args:
        all_oof_preds (Dict[str, Predictions]): Dictionary mapping model names to their out-of-fold predictions.
        all_test_preds (Dict[str, Predictions]): Dictionary mapping model names to their aggregated test predictions.
        y_true_full (Labels): Ground truth labels, available for evaluation and optimization.
    Returns:
        Predictions: Final test set predictions.
        
    Requirements:
      - Return a non-None value
      - Output must have the same number of samples as each fold's test predictions
      - Output must NOT contain NaN or Infinity values
      - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
    \"\"\"
    # Step 1: Evaluate individual model scores and prediction correlations
    # Step 2: Apply ensemble strategy
    # Step 3: Return final test predictions
    pass
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.

"""


class WorkflowPrompts:
    """Prompts for the 'workflow' stage."""
    SYSTEM = """
You are a world-class data scientist and machine learning engineer, and your current task is to act as a pipeline integrator.
You will be given a set of Python functions, each responsible for a specific stage of a machine learning process (data loading, feature engineering, training, ensembling).
Your job is to write a single `workflow` function that correctly calls these functions in sequence to execute the full end-to-end pipeline and produce artifacts required by the task description.

## Hardware Context
The following hardware resources are available.
<hardware_info>
{{hardware_info}}
</hardware_info>

## Task File Context
The following files are present in the directory `{{task_data_path}}`.
Use these exact paths and file sizes to plan your code.
<task_dir_structure>
{{task_dir_structure}}
</task_dir_structure>

## Task Description
<task_description>
{{task_description}}
</task_description>

## COMPONENT CONTEXT (Dependencies)
The following code blocks are the **IMMUTABLE** implementations of upstream pipeline components.
Your implementation MUST be fully compatible with them — respect their data structures, return formats, and behaviors exactly.

--------- load_data function ---------
File: load_data.py
<python_code>
{{load_data_code}}
</python_code>

--------- cross_validation function ---------
File: cross_validation.py
<python_code>
{{cross_validation_code}}
</python_code>

--------- create_features function ---------
File: create_features.py
<python_code>
{{feature_code}}
</python_code>

--------- train_and_predict function ---------
File: train_and_predict.py
<python_code>
{{model_code}}
</python_code>

--------- ensemble function ---------
File: ensemble.py
<python_code>
{{ensemble_code}}
</python_code>

## Guidelines
{% if gpu_available %}
0. A CUDA-enabled GPU is available - You MUST enable GPU acceleration wherever supported by the libraries you choose. Failure to utilize the available GPU is considered an error.
   **GPU Parameter Reference**: LightGBM uses `device='cuda'` (NOT `'gpu'`); XGBoost uses `device='cuda'`; CatBoost uses `task_type='GPU'` (uppercase). Always verify exact parameter names from official documentation.
{% endif %}
1. This workflow function will be executed in the production environment to generate final artifacts. It must process the COMPLETE dataset.
2. All file paths must be relative to: `{{task_data_path}}` for loading data, `{{output_data_path}}` for saving outputs.
3. Your task is ONLY to integrate the provided functions. Do NOT modify the logic within the component functions.
4. Ensure you import all necessary functions from their respective (hypothetical) modules.
5. Return ONLY the Python code implementation
6. Any error that could affect output quality must propagate immediately rather than attempting fallback workarounds — this ensures errors can be diagnosed and fixed.
"""
    USER = """
Please implement the Python code for the 'workflow' stage by integrating the functions provided in system prompt.

## IMPLEMENTATION GUIDANCE
{% if plan %}
{{plan}}
{% else %}
Execute a robust workflow pipeline.
{% endif %}

{% if parent_code %}
## PREVIOUS IMPLEMENTATION
**Evolution Task:** Modify following parent code to achieve the requirements above. Preserve working logic that doesn't conflict with the change.

<parent_code>
{{parent_code}}
</parent_code>
{% endif %}

## FUNCTION SPECIFICATION
Your code must implement the following `workflow` function:

<python_code>
# Assume all component functions are available for import
from load_data import load_data
from create_features import create_features
from cross_validation import cross_validation
from train_and_predict import PREDICTION_ENGINES
from ensemble import ensemble

BASE_DATA_PATH = "{{task_data_path}}"
OUTPUT_DATA_PATH = "{{output_data_path}}"

def workflow()->dict:
    \"\"\"
    Orchestrates the complete end-to-end machine learning pipeline in production mode.
    
    This function integrates all pipeline components (data loading, feature engineering, 
    cross-validation, model training, and ensembling) to generate final deliverables 
    specified in the task description.
    
    **IMPORTANT: This executes the PRODUCTION pipeline with the COMPLETE dataset.**
    Returns:
        dict: A dictionary containing all task deliverables.
              Required keys:
              - 'submission_file_path': Path to the final submission CSV
              - 'model_scores': Dict mapping model names to CV scores
              - 'prediction_stats': Prediction distribution statistics (see format below)
              
              Optional keys:
              - Additional task-specific metrics or file paths
    
    Requirements:
        - **MUST call `load_data(validation_mode=False)` to load the full dataset**
        - Return value must be JSON-serializable (primitive types, lists, dicts only)
        - Save any non-serializable objects (models, arrays, DataFrames) to files under `{{output_data_path}}`
        - Do not attempt fallback handling that could mask issues affecting output quality — let errors propagate
        
    prediction_stats Format:
        {
            "oof": {                    # Out-of-Fold prediction statistics
                "mean": float,          # Mean of OOF predictions
                "std": float,           # Standard deviation
                "min": float,           # Minimum value
                "max": float,           # Maximum value
            },
            "test": {                   # Test prediction statistics
                "mean": float,          # Mean of test predictions
                "std": float,           # Standard deviation
                "min": float,           # Minimum value
                "max": float,           # Maximum value
            }
        }
    \"\"\"
    # Your implementation goes here.
    # 1. Load full dataset with load_data(validation_mode=False)
    # 2. Set up cross-validation strategy
    # 3. For each fold:
    #      a. Split train/validation data
    #      b. Apply create_features() to this fold
    #      c. Train each model and collect OOF + test predictions
    # 4. Ensemble predictions from all models
    # 5. Compute prediction statistics
    # 6. Generate deliverables (submission file, scores, etc.)
    # 7. Save artifacts to files and return paths in a JSON-serializable dict
    # Example implementation for a task:
    output_info = {
        "submission_file_path": "path/to/submission.csv",
        "model_scores": {"model_name": 0.85},
        "prediction_stats": {
            "oof": {"mean": float(xx), "std": float(xx), "min":float(xx), "max":float(xx)},
            "test": {"mean": float(xx), "std": float(xx), "min":float(xx), "max":float(xx)},
        },
    }
    return output_info
</python_code>

## CRITICAL OUTPUT REQUIREMENT
Respond with ONLY the Python code block. No preamble, no explanation, no analysis.
Your response must start with <python_code> and end with </python_code>.

"""


class PackageInstallerPrompts:
    """Prompts for package installer"""
    USER = """
You are an expert package installer.
Your task is to provide the package installation command based on the error message.

# Error Message
{error_msg}

# Language
{language}

```bash
# The package installation command is here.
```
"""
