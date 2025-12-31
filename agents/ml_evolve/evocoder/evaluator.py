# -*- coding: utf-8 -*-
"""
This file define EvoCoderEvaluator
"""

import abc
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass

from agentsdk.logger.logger import get_logger
from evolux.evolve.context import EvaluatorConfig
from evolux.evolve.evaluator import LoongFlowEvaluator


@dataclass
class EvoCoderEvaluatorConfig:
    """
    Configuration class for EvoCoderEvaluator.
    """

    workspace_path: str = None
    timeout: int = 1800


class EvoCoderEvaluator(LoongFlowEvaluator, abc.ABC):
    """
    EvoCoder Evaluator
    """

    def __init__(self, config: EvoCoderEvaluatorConfig):
        cfg = EvaluatorConfig(
            workspace_path=config.workspace_path,
            timeout=config.timeout,
            evaluate_code=self._get_evaluate_code(),
        )
        super().__init__(cfg)

    @abc.abstractmethod
    def _get_evaluate_code(self) -> str:
        """
        get evaluation code
        """
        pass

    @staticmethod
    def _run_evaluate_target(evaluator_file_path: str, llm_file_path: str):
        """
        Run the evaluator code in a separate process and write results to a file.
        """
        base_dir = os.path.dirname(evaluator_file_path)
        output_file_path = os.path.join(base_dir, "evaluation_result.json")
        log_file_path = os.path.join(base_dir, "evaluation_process.log")

        log_fd = os.open(log_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log_fd, 1)  # stdout
        os.dup2(log_fd, 2)  # stderr
        os.close(log_fd)

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        log_file = open(log_file_path, "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file

        logger = get_logger("EvoCoderEvaluator_Child")
        pid = os.getpid()
        logger.debug(
            f"[Child PID:{pid}] Started. Evaluator path: {evaluator_file_path}"
        )
        logger.debug(f"[Child PID:{pid}] Target LLM path: {llm_file_path}")

        try:
            # 1. Dynamically load the LLM code module
            spec = importlib.util.spec_from_file_location(
                "llm_code_module", llm_file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Module spec creation failed for {llm_file_path}")

            llm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_module)
            # 2. Load EVOCODER_FILES variable
            if not hasattr(llm_module, "EVOCODER_FILES"):
                raise AttributeError("Invalid llm_code, EVOCODER_FILES not found")

            project_files = llm_module.EVOCODER_FILES
            if not isinstance(project_files, dict):
                raise TypeError("EVOCODER_FILES must be a dict")
            # 3. Write all project files to disk
            project_dir = os.path.dirname(llm_file_path)
            logger.debug(
                f"[Child PID:{pid}] Writing {len(project_files)} project files to {project_dir}"
            )

            for filename, content in project_files.items():
                full_path = os.path.join(project_dir, f"{filename}.py")
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # 4. Dynamically load evaluator code module
            spec = importlib.util.spec_from_file_location(
                "evaluator_code", evaluator_file_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Module spec creation failed for {evaluator_file_path}"
                )

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "evaluate"):
                raise AttributeError(
                    "The evaluator module must contain an 'evaluate' function."
                )
            # 5. Call evaluate with project directory
            start_time = time.time()
            result = mod.evaluate(project_dir)
            duration = time.time() - start_time
            logger.debug(f"[Child PID:{pid}] evaluate() finished in {duration:.4f}s.")
            if not isinstance(result, dict):
                raise TypeError(
                    f"The 'evaluate' function must return a dict, but got {type(result)}"
                )
            result_data = result

        except Exception as e:
            logger.error(f"[Child PID:{pid}] Exception occurred: {e}")
            logger.error(traceback.format_exc())
            result_data = {"error": str(e), "traceback": traceback.format_exc()}

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
        except Exception as write_err:
            logger.error(f"[Child PID:{pid}] Failed to write result file: {write_err}")

        logger.debug(f"[Child PID:{pid}] Process logic finished. Forcing exit.")

        try:
            log_file.flush()
            os.fsync(log_file.fileno())
            log_file.close()
        except:
            pass

        logger.debug(
            f"Subprocess exit now. Result score: {result_data.get('score', '')}. "
            f"Result summary: {result_data.get('summary', '')}"
        )
        os._exit(0)


COMMON_UTILS = """
def get_len(data):
    if data is None: return 0   
    if hasattr(data, 'shape'):
        shape = data.shape
        if not shape:  return 1
        return shape[0] 
    try: return len(data)
    except TypeError: return 1
    
def get_cols(data):
    if data is None: return 0
    
    if hasattr(data, 'shape'):
        shape = data.shape
        if not shape: return 1
        if len(shape) == 1: return 1
        return shape[1]
        
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            return len(data[0]) # List of List
        return 1 # List of Scalars
        
    return 1

def check_nan_inf_safe(data):
    import numpy as np
    try:
        data_type_str = str(type(data))

        # 1. PyTorch Tensor
        if 'torch' in data_type_str and 'Tensor' in data_type_str:
            import torch
            is_nan = torch.isnan(data).any().item()
            is_inf = torch.isinf(data).any().item()
            return is_nan or is_inf

        # 2. TensorFlow Tensor/Variable
        if 'tensorflow' in data_type_str and ('Tensor' in data_type_str or 'Variable' in data_type_str):
            import tensorflow as tf
            is_nan = tf.math.is_nan(data).numpy().any()
            is_inf = tf.math.is_inf(data).numpy().any()
            return is_nan or is_inf

        # 3. cuDF DataFrame/Series
        if 'cudf' in data_type_str and ('DataFrame' in data_type_str or 'Series' in data_type_str):
            import cupy as cp 
            arr = data.to_cupy() 
            return cp.isnan(arr).any().item() or cp.isinf(arr).any().item()

        # 4. CuPy ndarray
        if 'cupy' in data_type_str and 'ndarray' in data_type_str:
            import cupy as cp
            return cp.isnan(data).any().item() or cp.isinf(data).any().item()

        
        # 5. SciPy Sparse Matrix
        if hasattr(data, 'data') and hasattr(data, 'tocsr'):
            arr = data.data 
            return np.isnan(arr).any() or np.isinf(arr).any()

        # 6. NumPy/Pandas/List
        # np.array() solve Pandas DataFrame, NumPy array, and Python list
        arr = np.asarray(data)
        if arr.dtype.kind in 'iufc':
            return np.isnan(arr).any() or np.isinf(arr).any()

        return False

    except Exception:
        return False
"""


class EDAEvaluator(EvoCoderEvaluator):
    """evaluate eda code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import os, re, sys, traceback

def evaluate(temp_dir):
    report = None
    try:
        if temp_dir not in sys.path: sys.path.insert(0, temp_dir)
        import eda

        report = eda.eda()

        assert isinstance(report, str), "The eda() function must return a string."

        REQUIRED_PATTERNS = {{
            "Files": r"\*\*Files\*\*:",
            "Target": r"\*\*Target\*\*:",
            "Columns": r"\*\*Columns\*\*:",
            "Missing": r"\*\*Missing\*\*:",
        }}
        missing_keys = [k for k, p in REQUIRED_PATTERNS.items() if not re.search(p, report)]
        
                
        if missing_keys:
            return {{
                "score": 0.0, 
                "status": "validation_failed", 
                "summary": f"Missing keys (substring match): {{missing_keys}}. eda_info: {{report}}"
            }}

        return {{"score": 1.0, "status": "success", "summary": "EDA validation passed.", "artifacts": {{"eda_info": report}}}}
    except Exception as e:
        if report is None:
            report = "no report from eda function"
        return {{
            "score": 0.0, 
            "status": "validation_failed", 
            "summary": f"error info: {{e}}; eda_info() result: {{report}}", 
            "artifacts": {{"traceback": traceback.format_exc()}}
        }}
"""


class LoadDataEvaluator(EvoCoderEvaluator):
    """evaluate load data code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import sys,traceback

{COMMON_UTILS}


def evaluate(temp_dir):
    try:
        if 'load_data' in sys.modules: del sys.modules['load_data']
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)

        import load_data

        try:
            result = load_data.load_data(validation_mode=True)
        except Exception as e:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"load_data execution crashed: {{str(e)}}",
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}

        if not isinstance(result, (tuple, list)):
            return {{"score": 0.0, "status": "validation_failed",
                    "summary": f"Return value expected Tuple/List, got {{type(result)}}"}}

        if len(result) != 4:
            return {{"score": 0.0, "status": "validation_failed",
                    "summary": f"Expected 4 return values (X, y, X_test, test_ids), got {{len(result)}}"}}

        X, y, X_test, test_ids = result

        if get_len(X) == 0: return {{"score": 0.0, "status": "validation_failed",
                                    "summary": "X (Training Data) is empty."}}
        if get_len(y) == 0: return {{"score": 0.0, "status": "validation_failed", "summary": "y (Labels) is empty."}}
        if get_len(X_test) == 0: return {{"score": 0.0, "status": "validation_failed", "summary": "X_test is empty."}}

        len_X = get_len(X)
        len_y = get_len(y)
        if len_X != len_y:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Training sample mismatch: X has {{len_X}} rows, y has {{len_y}} rows."
            }}

        len_Xt = get_len(X_test)
        len_ids = get_len(test_ids)
        if len_Xt != len_ids:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Test sample mismatch: X_test has {{len_Xt}} rows, test_ids has {{len_ids}} rows."
            }}

        cols_X = get_cols(X)
        cols_Xt = get_cols(X_test)

        if cols_X != cols_Xt:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Feature dimension mismatch: Train has {{cols_X}} cols, Test has {{cols_Xt}} cols. Models "
                           f"will likely fail."
            }}

        if hasattr(X, 'columns') and hasattr(X_test, 'columns'):
            try:
                c_train = list(X.columns)
                c_test = list(X_test.columns)

                if c_train != c_test:
                    return {{
                        "score": 0.0,
                        "status": "validation_failed",
                        "summary": "Train/Test column names are same but in DIFFERENT ORDER."
                    }}
            except:
                pass

        return {{
            "score": 1.0,
            "status": "success",
            "summary": f"Data loaded successfully. Train: ({{len_X}}, {{cols_X}}), Test: ({{len_Xt}}, {{cols_Xt}})"
        }}

    except Exception as e:
        return {{
            "score": 0.0,
            "status": "validation_failed",
            "summary": f"Validation logic error: {{str(e)}}",
            "artifacts": {{"traceback": traceback.format_exc()}}
        }}

"""


# --- Stage 2: Cross Validation ---
class CrossValidationEvaluator(EvoCoderEvaluator):
    """evaluate cross validation code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import sys, traceback, numpy as np, pandas as pd
{COMMON_UTILS}

def evaluate(temp_dir):
    try:
        if 'cross_validation' in sys.modules: del sys.modules['cross_validation']
        if 'load_data' in sys.modules: del sys.modules['load_data']
            
        if temp_dir not in sys.path: sys.path.insert(0, temp_dir)
        import load_data
        import cross_validation

        X_sub, y_sub, _, _ = load_data.load_data(validation_mode=True)
        
        try:
            splitter = cross_validation.cross_validation(X_sub, y_sub)
        except Exception as e:
            return {{
                "score": 0.0, 
                "status": "validation_failed", 
                "summary": f"cross_validation instantiation failed: {{str(e)}}", 
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}

        if not hasattr(splitter, 'split'):
            return {{"score": 0.0, "status": "validation_failed", "summary": "Returned object missing 'split' method."}}
        
        if not hasattr(splitter, 'get_n_splits'):
            return {{"score": 0.0, "status": "validation_failed", "summary": "Returned object missing 'get_n_splits' method."}}

        try:
            generator = splitter.split(X_sub, y_sub)
            first_fold = next(generator, None)
            
            if first_fold is None:
                 return {{"score": 0.0, "status": "validation_failed", "summary": "Splitter produced no splits (generator is empty)."}}
            
            train_idx, val_idx = first_fold
            
            if not hasattr(train_idx, '__len__') or not hasattr(val_idx, '__len__'):
                return {{"score": 0.0, "status": "validation_failed", "summary": "Splitter must yield train/test indices."}}
                
            try:
                intersection = set(train_idx) & set(val_idx)
                if len(intersection) > 0:
                    return {{
                        "score": 0.0, 
                        "status": "validation_failed", 
                        "summary": f"Data Leakage detected! {{len(intersection)}} samples overlap between train and validation set."
                    }}
            except:
                pass

        except ValueError as ve:
            return {{
                "score": 0.0, 
                "status": "validation_failed", 
                "summary": f"Splitter incompatible with data logic: {{str(ve)}}. (Did you use StratifiedKFold on regression target?)",
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}
        except Exception as e:
            return {{
                "score": 0.0, 
                "status": "validation_failed", 
                "summary": f"Splitter.split() failed execution: {{str(e)}}",
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}

        return {{"score": 1.0, "status": "success", "summary": "cross_validation strategy is valid and executable."}}

    except Exception as e:
        return {{"score": 0.0, "status": "validation_failed", "summary": f"Validation logic error: {{str(e)}}", "artifacts": {{"traceback": traceback.format_exc()}}}}
"""


# --- Stage 3: Create Features ---
class CreateFeaturesEvaluator(EvoCoderEvaluator):
    """evaluate create features code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import sys,traceback,copy,pandas as pd,numpy as np

{COMMON_UTILS}

def evaluate(temp_dir):
    try:
        if 'create_features' in sys.modules: del sys.modules['create_features']
        if 'load_data' in sys.modules: del sys.modules['load_data']

        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)

        import load_data
        import create_features

        X_tr_in, y_tr_in, X_te_in, test_ids = load_data.load_data(validation_mode=True)

        try:
            result = create_features.create_features(X_tr_in, y_tr_in, X_tr_in, y_tr_in, X_te_in)
        except Exception as e:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"create_features execution crashed: {{str(e)}}",
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}

        if not isinstance(result, (tuple, list)) or len(result) != 5:
            return {{"score": 0.0, "status": "validation_failed",
                    "summary": f"Expected 5 return values (X_tr, y_tr, X_te), got {{type(result)}}"}}

        X_tr_out, y_tr_out, X_val_out, y_val_out, X_te_out = result

        if X_tr_out is None or X_te_out is None or X_val_out is None or y_tr_out is None or y_val_out is None:
            return {{"score": 0.0, "status": "validation_failed", "summary": "Output features cannot be None."}}

        if get_len(X_te_out) != get_len(X_te_in):
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Test set row count changed! Input: {{get_len(X_te_in)}}, Output: {{get_len(X_te_out)}}."
            }}

        if get_len(X_tr_out) != get_len(y_tr_out):
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Training X and y misaligned! X rows: {{get_len(X_tr_out)}}, y rows: {{get_len(y_tr_out)}}."
            }}
            
        if get_len(X_val_out) != get_len(y_val_out):
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Validation X and y misaligned! X rows: {{get_len(X_val_out)}}, y rows: {{get_len(y_val_out)}}."
            }}

        cols_tr = get_cols(X_tr_out)
        cols_te = get_cols(X_te_out)

        if cols_tr != cols_te:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Feature columns mismatch! Train has {{cols_tr}} cols, Test has {{cols_te}} cols."
            }}

        if check_nan_inf_safe(X_tr_out):
            return {{"score": 0.0, "status": "validation_failed", "summary": "Output training features contain NaN or Infinity."}}
            
        if check_nan_inf_safe(X_val_out):
            return {{"score": 0.0, "status": "validation_failed", "summary": "Output validation features contain NaN or Infinity."}}
                                                                             

        if hasattr(X_tr_out, 'columns') and hasattr(X_te_out, 'columns'):
            c_tr = list(X_tr_out.columns)
            c_te = list(X_te_out.columns)

            if set(c_tr) != set(c_te):
                return {{"score": 0.0, "status": "validation_failed", "summary": "Train/Test column names set mismatch."}}

        return {{"score": 1.0, "status": "success",
                "summary": f"Features valid. Shape: ({{get_len(X_tr_out)}}, {{get_cols(X_tr_out)}})"}}

    except Exception as e:
        return {{
            "score": 0.0,
            "status": "validation_failed",
            "summary": f"Validation logic error: {{str(e)}}",
            "artifacts": {{"traceback": traceback.format_exc()}}
        }}
"""


# --- Stage 4: Train and Predict ---
class TrainAndPredictEvaluator(EvoCoderEvaluator):
    """evaluate train and predict code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import numpy as np,sys,traceback

{COMMON_UTILS}

def evaluate(temp_dir):
    try:
        keys_to_remove = ['train_and_predict', 'create_features', 'load_data']
        for key in keys_to_remove:
            if key in sys.modules:
                del sys.modules[key]

        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)

        import load_data
        import create_features
        import train_and_predict

        if not hasattr(train_and_predict, 'PREDICTION_ENGINES'):
            return {{"score": 0.0, "status": "validation_failed",
                    "summary": "Module missing 'PREDICTION_ENGINES' dictionary."}}

        engines = train_and_predict.PREDICTION_ENGINES
        if not engines:
            return {{"score": 0.0, "status": "validation_failed", "summary": "PREDICTION_ENGINES is empty."}}

        X_sub, y_sub, X_test_sub, _ = load_data.load_data(validation_mode=True)
        X_feat, y_feat, X_val, y_val, X_test_feat = create_features.create_features(X_sub, y_sub, X_sub, y_sub, X_test_sub)

        X_tr = X_feat
        y_tr = y_feat
        len_val = get_len(y_val)
        len_test = get_len(X_test_feat)

        success_count = 0
        errors = []

        for name, train_func in engines.items():
            try:
                oof_preds, test_preds = train_func(
                    X_tr, y_tr, X_val, y_val, X_test_feat
                )

                if oof_preds is None or test_preds is None:
                    errors.append(f"Model '{{name}}' returned None.")
                    continue

                if get_len(oof_preds) != len_val:
                    errors.append(f"Model '{{name}}' OOF shape mismatch: expected {{len_val}}, got {{get_len(oof_preds)}}")
                    continue

                if get_len(test_preds) != len_test:
                    errors.append(f"Model '{{name}}' Test shape mismatch: expected {{len_test}}, got {{get_len(test_preds)}}")
                    continue

                if check_nan_inf_safe(oof_preds):
                    errors.append(f"Model '{{name}}' produced NaNs or Infs in validation predictions.")
                    continue

                success_count += 1

            except Exception as model_e:
                errors.append(f"Model '{{name}}' crashed: {{str(model_e)}}")

        total_engines = len(engines)
        if success_count == total_engines:
            return {{"score": 1.0, "status": "success", "summary": f"All {{success_count}} models passed validation."}}
        else:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Only {{success_count}}/{{total_engines}} passed. Errors: {{'; '.join(errors)}}"
            }}

    except Exception as e:
        return {{
            "score": 0.0,
            "status": "validation_failed",
            "summary": f"Validation logic internal error: {{str(e)}}",
            "artifacts": {{"traceback": traceback.format_exc()}}
        }}
"""


# --- Stage 5: Ensemble ---
class EnsembleEvaluator(EvoCoderEvaluator):
    """evaluate ensemble code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import sys, traceback, numpy as np, pandas as pd

{COMMON_UTILS}

def evaluate(temp_dir):
    try:
        keys_to_remove = ['ensemble', 'train_and_predict', 'create_features', 'load_data']
        for key in keys_to_remove:
            if key in sys.modules:
                del sys.modules[key]
        if temp_dir not in sys.path:
            sys.path.insert(0, temp_dir)
            
        import load_data
        import create_features
        import train_and_predict
        import ensemble
        
        X_sub, y_sub, X_test_sub, _ = load_data.load_data(validation_mode=True)

        X_feat, y_feat, X_val, y_val, X_test_feat = create_features.create_features(
            X_sub, y_sub, X_sub, y_sub, X_test_sub
        )

        X_tr = X_feat
        y_tr = y_feat
        N_TEST = get_len(X_test_feat)

        engines = train_and_predict.PREDICTION_ENGINES
        
        all_oof_preds = {{}}
        all_test_preds = {{}}
        for model_name, train_func in engines.items():
            try:
                oof_preds, test_preds = train_func(
                    X_tr, y_tr, X_val, y_val, X_test_feat
                )
                
                if oof_preds is not None and test_preds is not None:
                    all_oof_preds[model_name] = oof_preds
                    all_test_preds[model_name] = test_preds
                    
            except Exception as e:
                continue
        if not all_oof_preds or not all_test_preds:
            return {{"score": 0.0, "status": "validation_failed",
                    "summary": "No valid predictions from any model."}}

        try:
            final_pred = ensemble.ensemble(all_oof_preds, all_test_preds, y_feat)
        except Exception as e:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"ensemble execution crashed: {{str(e)}}",
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}

        if final_pred is None:
            return {{"score": 0.0, "status": "validation_failed", "summary": "Ensemble returned None."}}

        len_out = get_len(final_pred)
        if len_out != N_TEST:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Ensemble output length mismatch. Expected {{N_TEST}} (test set size), got {{len_out}}."
            }}

        if check_nan_inf_safe(final_pred):
            return {{"score": 0.0, "status": "validation_failed", "summary": "Ensemble output contains NaN or "
                                                                            "Infinity."}}

        return {{"score": 1.0, "status": "success", "summary": "Ensemble logic validated successfully."}}

    except Exception as e:
        return {{
            "score": 0.0,
            "status": "validation_failed",
            "summary": f"Validation logic error: {{str(e)}}",
            "artifacts": {{"traceback": traceback.format_exc()}}
        }}

"""


# --- Stage 6: Workflow ---
class WorkflowEvaluator(EvoCoderEvaluator):
    """evaluate workflow code"""

    def _get_evaluate_code(self) -> str:
        return f"""
import os, sys, time, pandas as pd, traceback

def evaluate(temp_dir):
    try:
        if temp_dir not in sys.path: sys.path.insert(0, temp_dir)
        # import all modules
        import load_data
        import cross_validation
        import create_features
        import train_and_predict
        import ensemble
        import workflow

        # check if workflow run successfully
        try:
            result = workflow.workflow()
        except Exception as e:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"workflow execution crashed: {{str(e)}}",
                "artifacts": {{"traceback": traceback.format_exc()}}
            }}

        assert isinstance(result, dict), "The workflow() function must return a dict."
        
        # check required keys
        REQUIRED_KEYS = ["submission_file_path", "model_scores", "prediction_stats"]
        missing_keys = [k for k in REQUIRED_KEYS if k not in result]
        if missing_keys:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Missing required keys in workflow result: {{missing_keys}}",
                "artifacts": {{"workflow": result}}
            }}
        # check prediction_stats structure
        prediction_stats = result["prediction_stats"]
        assert isinstance(prediction_stats, dict), "prediction_stats must be a dict."
        
        REQUIRED_STAT_KEYS = ["oof", "test"]
        missing_stat_keys = [k for k in REQUIRED_STAT_KEYS if k not in prediction_stats]
        if missing_stat_keys:
            return {{
                "score": 0.0,
                "status": "validation_failed",
                "summary": f"Missing keys in prediction_stats: {{missing_stat_keys}}",
                "artifacts": {{"workflow": result}}
            }}
        # check oof and test structure
        REQUIRED_FIELDS = ["mean", "std", "min", "max"]
        for stat_key in REQUIRED_STAT_KEYS:
            stat_obj = prediction_stats[stat_key]
            assert isinstance(stat_obj, dict), f"prediction_stats[{{stat_key}}] must be a dict."
            
            missing_fields = [f for f in REQUIRED_FIELDS if f not in stat_obj]
            if missing_fields:
                return {{
                    "score": 0.0,
                    "status": "validation_failed",
                    "summary": f"Missing fields in prediction_stats[{{stat_key}}]: {{missing_fields}}",
                    "artifacts": {{"workflow": result}}
                }}
            
            # check field types are numeric
            for field in REQUIRED_FIELDS:
                value = stat_obj[field]
                if not isinstance(value, (int, float)):
                    return {{
                        "score": 0.0,
                        "status": "validation_failed",
                        "summary": f"prediction_stats[{{stat_key}}][{{field}}] must be numeric, got {{type(value).__name__}}",
                        "artifacts": {{"workflow": result}}
                    }}
        return {{
            "score": 1.0, 
            "status": "success", 
            "summary": "workflow validation passed and submission file created.",
            "artifacts": {{"workflow": result}}
        }}
        
    except Exception as e:
        return {{"score": 0.0, "status": "validation_failed", "summary": f"{{e}}", "artifacts": {{"traceback": traceback.format_exc()}}}}
"""
