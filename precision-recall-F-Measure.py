"""
This code evaluates the performance of two binary classification models, Model M1 and Model M2, by computing three key metrics: Precision, Recall, and F-measure. These metrics are calculated based on a threshold, which determines whether the model's predicted probability is classified as 0 or 1.

Precision is the proportion of true positive predictions out of all positive predictions made by the model.
Recall is the proportion of true positive predictions out of all actual positives in the data.
F-measure (or F1-score) is the harmonic mean of Precision and Recall, providing a single metric that balances the two.

The code first computes these metrics for Model M1 and Model M2 using a default threshold of 0.5. Then, it recalculates the metrics for Model M1 using a new threshold of 0.1 to observe how the model's performance changes with the threshold adjustment.
"""

import numpy as np

# Set the initial threshold
threshold = 0.5

# Generate binary predictions for Model M1 based on the threshold
predictions_M1 = [1 if p > threshold else 0 for p in probabilities_M1]

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for Model M1
TP = sum([1 for p, t in zip(predictions_M1, true_classes_binary) if p == 1 and t == 1])
FP = sum([1 for p, t in zip(predictions_M1, true_classes_binary) if p == 1 and t == 0])
FN = sum([1 for p, t in zip(predictions_M1, true_classes_binary) if p == 0 and t == 1])

# Compute Precision, Recall, and F-measure for Model M1
precision = TP / (TP + FP) if TP + FP > 0 else 0  # Handle division by zero
recall = TP / (TP + FN) if TP + FN > 0 else 0  # Handle division by zero
f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0  # Handle division by zero

# Output the metrics for Model M1
precision, recall, f_measure

# Generate binary predictions for Model M2 based on the same threshold
predictions_M2 = [1 if p > threshold else 0 for p in probabilities_M2]

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for Model M2
TP_M2 = sum([1 for p, t in zip(predictions_M2, true_classes_binary) if p == 1 and t == 1])
FP_M2 = sum([1 for p, t in zip(predictions_M2, true_classes_binary) if p == 1 and t == 0])
FN_M2 = sum([1 for p, t in zip(predictions_M2, true_classes_binary) if p == 0 and t == 1])

# Compute Precision, Recall, and F-measure for Model M2
precision_M2 = TP_M2 / (TP_M2 + FP_M2) if TP_M2 + FP_M2 > 0 else 0  # Handle division by zero
recall_M2 = TP_M2 / (TP_M2 + FN_M2) if TP_M2 + FN_M2 > 0 else 0  # Handle division by zero
f_measure_M2 = 2 * (precision_M2 * recall_M2) / (precision_M2 + recall_M2) if precision_M2 + recall_M2 > 0 else 0  # Handle division by zero

# Output the metrics for Model M2
precision_M2, recall_M2, f_measure_M2

# Set a new threshold for Model M1
threshold_new = 0.1

# Generate binary predictions for Model M1 using the new threshold
predictions_M1_new_threshold = [1 if p > threshold_new else 0 for p in probabilities_M1]

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) for Model M1 with the new threshold
TP_M1_new = sum([1 for p, t in zip(predictions_M1_new_threshold, true_classes_binary) if p == 1 and t == 1])
FP_M1_new = sum([1 for p, t in zip(predictions_M1_new_threshold, true_classes_binary) if p == 1 and t == 0])
FN_M1_new = sum([1 for p, t in zip(predictions_M1_new_threshold, true_classes_binary) if p == 0 and t == 1])

# Compute Precision, Recall, and F-measure for Model M1 with the new threshold
precision_M1_new = TP_M1_new / (TP_M1_new + FP_M1_new) if TP_M1_new + FP_M1_new > 0 else 0  # Handle division by zero
recall_M1_new = TP_M1_new / (TP_M1_new + FN_M1_new) if TP_M1_new + FN_M1_new > 0 else 0  # Handle division by zero
f_measure_M1_new = 2 * (precision_M1_new * recall_M1_new) / (precision_M1_new + recall_M1_new) if precision_M1_new + recall_M1_new > 0 else 0  # Handle division by zero

# Output the metrics for Model M1 with the new threshold
precision_M1_new, recall_M1_new, f_measure_M1_new
