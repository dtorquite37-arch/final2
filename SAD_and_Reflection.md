# Student Performance Prediction System — SAD Mini-Analysis, Test Evidence, and Reflection

**Name:** ____________________  **Section:** ____________  **Date:** ____________
**GitHub repository URL:** https://github.com/YOUR-USERNAME/student-performance-prediction

## 1. SAD mini-analysis

| SAD element | Response |
|---|---|
| **Problem statement** | Academic risk is often noticed only after poor results. The prototype explores early identification of students at risk of failing final Mathematics (G3 < 10) using available indicators. |
| **Users** | Faculty member or academic analyst. |
| **Inputs** | Student attributes and academic indicators (e.g., study time, absences, past failures, G1, G2) supplied as a CSV file. |
| **Process** | Validate required columns → preprocess (impute, scale, one-hot encode) → apply trained Random Forest → classify risk. |
| **Outputs** | AT RISK / NOT AT RISK label per record, downloadable results CSV, and a model evaluation summary. |
| **Key limitation** | A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population. |

**Use case:** The user uploads a CSV → the system checks required columns → the model predicts risk → the system displays and exports the results.

## 2. Model evaluation results
*(Copy from `model_evaluation.txt` after running `python train_model.py` on the real dataset.)*

| Metric | Value |
|---|---|
| Accuracy | |
| Precision (At risk) | |
| Recall (At risk) | |
| F1-score (At risk) | |

## 3. Test evidence

| Test | Expected result | Pass? |
|---|---|---|
| Run `train_model.py` | CSV and model files are created without errors | ☐ |
| Run Streamlit app | Dashboard loads in browser | ☐ |
| Upload sample CSV | Predictions are displayed | ☐ |
| Upload CSV missing a required column | App displays a helpful error message | ☐ |
| Download results | CSV containing predictions is downloaded | ☐ |
| Open GitHub repository | Source files and README are visible | ☐ |

*Tip for the missing-column test: open the sample CSV, delete the `absences` column, save as `bad_input.csv`, and upload it.*

## 4. Reflection

**1. What is the prediction target, and why is G3 excluded from the input features?**
The target is whether a student is at risk of failing final Mathematics: 1 (AT RISK) if G3 < 10, otherwise 0 (NOT AT RISK). G3 is excluded because the target is computed directly from it. Including it would be target leakage: the model would simply read the answer, score near-perfectly, and be useless in practice, since the real final grade is unknown when we want to predict.

**2. What does the model's recall for the "At risk" class tell you?**
Recall is the share of truly at-risk students that the model correctly flags (TP / (TP + FN)). A recall of, say, 0.85 means 85% of at-risk students were caught and 15% were missed (false negatives). Because missing a student who needs help is costly, recall for this class matters more than overall accuracy, which can look high just because most students are not at risk.

**3. Why might a model trained on this dataset perform differently on students from your institution?**
The data come from two Portuguese secondary schools in a specific period. Differences in grading scale and policies, curriculum, culture, socioeconomic conditions, and student demographics change the relationships between features and outcomes (distribution shift). The dataset is also small (395 records), so the model may be overfitted to local patterns. It should be validated and retrained on local data before any use.

**4. What action should a faculty member take after seeing an "AT RISK" prediction?**
Treat it as a prompt for a supportive follow-up, not a verdict. Review the student's actual record and context, talk with the student, and consider tutoring, consultation, or counseling referrals. The prediction must not be the sole basis for decisions such as grading, labeling, or exclusion, and the faculty member should document and monitor outcomes.

## 5. Submission checklist
1. ☐ SAD mini-analysis (this document, Section 1)
2. ☐ `train_model.py` and `app.py`
3. ☐ `requirements.txt` and `README.md`
4. ☐ Model evaluation results (Section 2)
5. ☐ Screenshot of the Streamlit dashboard and prediction results
6. ☐ GitHub repository URL
