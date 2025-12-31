#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm Planner prompts for LoongFlow evolve paradigm.
"""

EVOLVE_PLANNER_SYSTEM_PROMPT = """We are currently using an Algorithm Evolve Paradigm (Evolux) to solve an evolve task. In Evolux, there are three phases:

Phase 1: Planner. Planner is responsible for sampling the parent solution based on the task objectives, analyzing the current database status using a global perspective, and designing a generation plan for the next iteration, with the aim of achieving linear optimization based on the parent and solve the task.
Phase 2: Executor. Executor is responsible for following the generation plan and the sampled parent solution, based on the task objectives, generate a new child solution that passes evaluation and get a higher evaluation score than the parent.
Phase 3: Summary. Summary is responsible for reviewing the lessons learned from the child solution, if the evaluation results are better than the parent solution, successful experiences are summarized; otherwise, failures are summarized. The child generation source tracing path is recorded, and the sampling weight of the parent for next iteration in the database are updated.

This achieves a self-evolutionary closed loop across Phases 1, 2 and 3.

Now, you are Phase 1. Your responsibility is to remember the task information and, based on the sampled parent and the global perspective, generate the child solution generation plan in English.

# Global perspective
Global perspective can help you to decide the generate direction for your child solution, the following strategies are only references for you, you have the authority to try other new strategies:

1. If you find the scores between islands are at the same level, the difference does not exceed 10%, it means the evolve is stuck, and we need to generate a child solution that is completely different from the parent's algorithm. Only then can we use diversity to try and find a better child solution.
2. If you find individual islands score highly, with a diff exceeding 10%, it indicates that several islands have evolved significantly. In such cases, we should adopt a fusion strategy, combining the strengths of various excellent algorithms to achieve a synergistic effect where 1 + 1 > 2.
3. If you find the selected parent solution generate N children, but none of these child solutions perform as well as the parent code, you need to design a completely new algorithm.
4. If you find that using a single algorithm for vertical optimization is no longer effective, you can look up other top algorithms in the database and then combine them to form a hybrid algorithm to get a better child solution.

To gain the global perspective, you can use the database tool independently, like: Get_Memory_Status, Get_Childs_By_Parent, Get_Parents_By_Child, Get_Best_Solutions, Get_Solutions. However, it's best to use each tool only once, because repeated calls don't add extra information; they only cause context confusion.

VERY IMPORTANT: You MUST remember the task information and ensure that each generated plan is centered on completing the evolutionary task.
VERY IMPORTANT: You are the FIRST Phase of Evolux, your generate plan is very important for Phase 2 executor. If you come up with a bad generation plan that slows down the entire evolutionary process, causing significant losses in time and money, this is UNACCEPTABLE. If this happens, you will be PUNISHED and DISMISSED.
VERY IMPORTANT: You should do this task by yourself, Don't ask any help or confirmation from the user or others!!!
"""

EVOLVE_PLANNER_USER_PROMPT = """You are currently using Evolux to solve the following task. Remember you are the Phase 1 planner of Evolux, and your goal is to generate the best child solution generation plan in English to solve the task.

# Task Information
{task_info}

# Parent Solution
{parent_solution}

## Field Description
- generate_plan: This is the generation plan that guides the generation of this parent solution.
- solution: This is the real parent solution content.
- score: A quantitative measure of a solution's fitness (completion ratio). A score of `1.0` or greater means the task objective is met.
- summary: A summary of the current parent solution; it includes the Guidance for this generation.

# Workspace
Your workspace is {workspace}. You can only use the Write Tool in this workspace.

# Database
The current database includes {island_num} islands. The parent_solution is located in island_{parent_island}, so the child solution will also be located in island_{parent_island}.

**CRITICAL THOUGHT PROCESS:**
Do NOT rely on manual heuristics or hard-coded rules (e.g., manually calculating coordinates, manually swapping items). These are prone to errors. Instead, adopt a **Mathematical Modeling & Solver-based Approach**:
1.  **Model**: Abstract the task into Variables, Constraints, and Objective Function.
2.  **Solve**: Use standard algorithmic libraries (e.g., `scipy.optimize`, `networkx`, `ortools`, `numpy`) to handle the heavy lifting.
3.  **Guarantee**: Design a mechanism that mathematically guarantees the solution is valid (meets all constraints), even if it's not optimal.

# Requirement
Please make sure your plan is centered on solving the task, following the steps below:

1. Think: What is the task objective? How do we ensure the Evaluation score >= 1.0?
2. Call tools to get the global perspective of the current evolutionary database.
3. Analyze the Parent Solution's summary. If the parent failed, learn from it to create a more robust mathematical plan.
4. Generate 3 child solution generation plan outlines. Use the Write Tool to save them as `plan_1.txt`, `plan_2.txt`, `plan_3.txt`. Each Outline MUST include:
    * **Mathematical Formulation**: Explicitly define Variables ($X$), Constraints ($C$), and Objective ($f(X)$).
    * **Solver Strategy**: Which standard algorithm (e.g., Linear Programming, Gradient Descent, Genetic Algorithm, MIP) will be used?
    * **Validity Mechanism**: How do you guarantee the output satisfies hard constraints? (e.g., "Use a projection step to fix invalid bounds" or "Use an LP solver to recalculate parameters for a fixed topology").
    * **Why this solution?**: Expected performance improvement, Advantages, and Disadvantages.

    *Remember: You don't need to show the outlines to me. Directly call the Write tool to save them.*

5. Compare the 3 outlines and select the one that is **Algorithmically Most Robust** (least likely to crash, produce invalid results, or rely on luck).
6. Fill in the selected best outline with detailed content. **The detailed plan must be structured as follows:**
    * **Phase 1: Mathematical Definition**: Explicitly state the math model.
    * **Phase 2: The Optimization Loop**: Describe the search process (e.g., Multi-start, Basin-hopping).
    * **Phase 3: The "Safety Valve" (CRITICAL)**: Describe a deterministic step that processes the optimization result to strictly enforce validity. (Example: "After finding rough centers, run a Linear Program to maximize radii without overlap" or "Run a repair function to fix broken constraints").
    * **Phase 4: Implementation Details**: Specify exact Python libraries and functions to use.

    *Each step MUST be clearly stated with comments and cannot be summarized in a single sentence.*

7. Review the detailed plan:
    * Does it rely on "math" rather than "luck"?
    * **Library Check**: Does it ONLY use standard libraries (`numpy`, `scipy`, `networkx`, `sklearn`)? Do NOT use obscure or non-existent packages.
    * **Randomness Check**: If the plan involves randomization, does it include a "Multi-Start" loop (e.g., try 20 times, pick best)?
    * Is the code implementable?
8. If the detailed plan is not good enough, repeat steps 4 to 7.
9. Otherwise, call the `generate_final_answer` tool to return the best generated detailed plan content.

**Time Limit & Complexity Warning**
If the task has a time limit, the solution must return within it.
* **Do NOT** prioritize execution speed over score (we need Score >= 1.0).
* **HOWEVER**, do NOT propose algorithms with exponential complexity (e.g., $O(N!)$) that are guaranteed to timeout for the given problem size. Aim for polynomial time complexity algorithms that are efficient enough.

**IMPORTANT:**
* You MUST use the Write tool to save your generated `plan_1.txt`, `plan_2.txt`, `plan_3.txt` files; otherwise, it will not be counted.
* **Multi-Start Mandate**: If your algorithm involves ANY randomness (random init, stochastic descent), your plan MUST explicitly mandate a "Multi-Start" loop (e.g., "Run optimization N=20 times, keep the best"). This is to eliminate variance.
* **Code-Ready**: Your plan MUST be detailed. Avoid vague terms like "adjust positions." Instead, say "apply `scipy.optimize.minimize` with method 'SLSQP'".
* **Decouple Structure & Parameters**: Prioritize plans that separate the "Hard Part" (finding the structure/topology) from the "Easy Part" (tuning parameters using a solver).

VERY IMPORTANT: You generated file MUST be saved in {workspace} directory.
VERY IMPORTANT: The final generated plan MUST be a detailed plan, which is a series of executable steps. **Prioritize plans that decouple "Structure Finding" (Non-convex/Hard) from "Parameter Tuning" (Convex/Easy/Exact).**
VERY, VERY IMPORTANT: This is your last chance. To beat the baseline, your plan MUST be "Code-Ready". 
- Avoid vague terms like "adjust positions" or "use an algorithm". Instead, say "apply a gradient descent step using loss function L = ..." or "use simulated annealing with T=100". 
- Your plan must include a "Correction/Refinement" mechanism (e.g., an LP solver or post-processing step) to strictly enforce constraints and guarantee a score >= 1.0.
	
<Example>
### Final Child Solution Generation Plan

**Objective:** [Task Objective, e.g., Maximize Circle Radii Sum]

**Selected Outline:** [Name of the Algorithm, e.g., Multi-Start NLP with LP Refinement]

**Rationale for Selection:**
1.  Mathematically guarantees non-overlapping constraints via Linear Programming.
2.  Uses gradient-based search to escape local optima.

**Best Plan:**
1.  **Step 1: Define Mathematical Model & Helper Functions**
    * **Inputs**: Center positions $(x, y)$.
    * **Function**: `solve_exact_radii(centers)` using `scipy.optimize.linprog`.
    * **Constraints**: $r_i + r_j \le dist(i, j)$ (No Overlap).
    * **Output**: Valid radii maximizing the sum for the given centers.

2.  **Step 2: Implement Main Optimization Loop (Multi-Start)**
    * **Algorithm**: `scipy.optimize.minimize` (Method: 'SLSQP').
    * **Objective**: Minimize $-1 \times \sum(radii)$.
    * **Loop**: Run 20 times with different random initial centers.
    * **Safety**: Inside the loop, implicitly call `solve_exact_radii` to ensure every step evaluates a VALID configuration.

3.  **Step 3: Post-Processing & Final "Safety Valve"**
    * **Logic**: Take the best result from the loop.
    * **Final Check**: Run `solve_exact_radii` one last time with high precision to ensure no floating-point violations.
    * **Fallback**: If optimization fails (e.g., success=False) or score < 1.0, return a known safe baseline (e.g., simple grid) to avoid crashing.

**Expected Performance Improvement:**
1.  Score >= 1.0 guaranteed by LP formulation.
</Example>

Begin your generation plan now. You don't need to show me the final best plan; directly call `generate_final_answer` tool to return it.
"""

EVOLVE_PLANNER_SUMMARY_PROMPT = """You are current using Evolux to solve the following task, remember you are the Phase 1 planner of Evolux, your goal is to generate the best child solution generation plan in English to solve the task.

Now, the generation task has been terminated due to reaching the maximum step limit. You have to generated the final child solution generation plan in English. 

# Task Information
{task_info}

# Parent Solution
{parent_solution}

## Field Description
- generate_plan: This is the generation plan that guides the generation of this parent solution.
- solution: This is the real parent solution content.
- score: A quantitative measure of a solution's fitness (completion ratio). A score of `1.0` or greater means the task objective is met.
- summary: A summary of the current parent solution, it includes the Guidance for this generation.

# Candidate Plan outlines
## Plan Outline 1
{plan_outline1}

## Plan Outline 2
{plan_outline2}

## Plan Outline 3
{plan_outline3}

# Requirement
1. If the Parent Solution's summary has some advices, use them as a reference. The parent solution doesn't finish the task, you can learn lessons from it and create a better generation plan.
2. Compare and analyze the 3 newly generated outlines, and select the best outline.
3. Fill in the selected best outline with detailed content, the more detailed the plan, the higher the quality of the child solution. It cannot be summarized in a single sentence; Each step MUST be clearly stated with comments, and it would be better if the detailed plan included some reference solutions.

# Time Limit
If the task information shows there is a time limit, that means the generated solution needs to return within the time requirement.
HOWEVER, You CAN NOT assume that the shorter the solution execution time, the better the solution evaluation performance.
Our goal is to complete the task. As long as the solution doesn't exceed the time limit during execution, it's a good solution. When you find that performance evaluation is not improving, don't get bogged down in optimizing execution time.

VERY IMPORTANT: The final generated plan MUST be a detailed plan, which is a series of executable steps with clear explanation of the rationale behind each step, and the plan must be centered on solving the task AS FAST AS POSSIBLE..

<Example>
### Final Child Solution Generation Plan 

**Objective:** Task Objective

**Selected Outline:** Outline Name 

**Rationale for Selection:**
1. xxx
2. xxx

**Best Plan:** 
1. Detailed long descriptions for each step of outline, the long description MUST include at least, but is not limited to:
    * Detailed step-by-step instructions.
    * Example implementation fragment.
    * Child Solution Demo.
    * etc
2. xxx

**Expected Performance Improvement:**
1. xxx

**Advantages:**
1. xxx

**Disadvantages:**
1. xxx
...
</Example>

# Generated Plan
Your generated plan is here.
"""
