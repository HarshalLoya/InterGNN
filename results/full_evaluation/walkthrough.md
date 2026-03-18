# InterGNN Evaluation Walkthrough

I have enhanced the InterGNN codebase by implementing a comprehensive evaluation pipeline in [run_full_evaluation.py](file:///c:/Users/harsh/Desktop/InterGNN/run_full_evaluation.py). This script produces all the requested performance metrics, baseline comparisons, interpretability evaluations, and statistical tests.

## Key Deliverables

### 1. Performance Comparison
InterGNN was evaluated against GCN and GIN baseline models on MUTAG and Tox21.

![Performance Comparison MUTAG](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/performance_comparison_mutag.png)
![Performance Comparison Tox21](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/performance_comparison_tox21.png)

### 2. Interpretability Evaluation
We assessed faithfulness (Deletion/Insertion AUC), stability (Jaccard), and chemical validity of the explanations.

![Interpretability Table MUTAG](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/interpretability_table_mutag.png)
![Interpretability Table Tox21](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/interpretability_table_tox21.png)

### 3. Generalization Tests
We compared performance across random and scaffold splits to measure how well the model generalizes to out-of-distribution scaffolds.

![Generalization Test Tox21](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/generalization_test_tox21.png)

### 4. Activity Cliff Visualization
We identified and visualized activity cliffs—structurally similar molecules with significant differences in activity—along with their atom importance explanations.

![Activity Cliff Tox21](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/activity_cliff_tox21.png)

### 5. Hypothesis Testing
We performed a formal paired t-test comparing the predictive performance of InterGNN and the GCN baseline.

![Hypothesis Testing Tox21](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/plots/hypothesis_testing_tox21.png)

## Generated Files
- **Plots**: `results/full_evaluation/plots/*.png`
- **Tables**: `results/full_evaluation/tables/*.csv`
- **LaTeX**: [results/full_evaluation/latex_tables.tex](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/latex_tables.tex)
- **Report**: [results/full_evaluation/report.txt](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/report.txt)
- **Summary**: [results/full_evaluation/results_summary.json](file:///c:/Users/harsh/Desktop/InterGNN/results/full_evaluation/results_summary.json)

You can run the full evaluation again (with more epochs if desired) using:
```bash
python run_full_evaluation.py --datasets mutag tox21
```
