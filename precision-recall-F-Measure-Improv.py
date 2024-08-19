import numpy as np

# Function to calculate precision, recall, and F-measure
def calculate_metrics(predictions, true_classes):
    """
    Calculate precision, recall, and F-measure based on predictions and true classes.
    
    Parameters:
    - predictions: list of binary predictions (0 or 1).
    - true_classes: list of actual binary classes (0 or 1).

    Returns:
    - precision: float, the precision metric.
    - recall: float, the recall metric.
    - f_measure: float, the F-measure metric.
    """
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = sum([1 for p, t in zip(predictions, true_classes) if p == 1 and t == 1])
    FP = sum([1 for p, t in zip(predictions, true_classes) if p == 1 and t == 0])
    FN = sum([1 for p, t in zip(predictions, true_classes) if p == 0 and t == 1])

    # Calculate Precision, Recall, and F-measure
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_measure

# Function to generate predictions based on a given threshold
def generate_predictions(probabilities, threshold):
    """
    Generate binary predictions based on a given threshold.

    Parameters:
    - probabilities: list of float values representing predicted probabilities.
    - threshold: float, the cutoff threshold for classification.

    Returns:
    - predictions: list of binary predictions (0 or 1).
    """
    return [1 if p > threshold else 0 for p in probabilities]

# Function to validate the input data
def validate_input(probabilities, true_classes):
    """
    Validate that the input probabilities and true classes are of the same length.

    Parameters:
    - probabilities: list of float values representing predicted probabilities.
    - true_classes: list of actual binary classes (0 or 1).

    Returns:
    - bool: True if inputs are valid, otherwise raises ValueError.
    """
    if len(probabilities) != len(true_classes):
        raise ValueError("The length of probabilities and true_classes must be the same.")
    return True

# Main evaluation process for Model M1 and M2
def evaluate_models(probabilities_M1, probabilities_M2, true_classes_binary, thresholds=[0.5, 0.1]):
    """
    Evaluate precision, recall, and F-measure for Model M1 and Model M2 at different thresholds.

    Parameters:
    - probabilities_M1: list of float values representing Model M1's predicted probabilities.
    - probabilities_M2: list of float values representing Model M2's predicted probabilities.
    - true_classes_binary: list of actual binary classes (0 or 1).
    - thresholds: list of float values representing thresholds to evaluate.

    Returns:
    - results: dict containing metrics for each model at each threshold.
    """
    # Validate input data
    validate_input(probabilities_M1, true_classes_binary)
    validate_input(probabilities_M2, true_classes_binary)

    results = {}
    
    for threshold in thresholds:
        # Evaluate Model M1
        predictions_M1 = generate_predictions(probabilities_M1, threshold)
        precision_M1, recall_M1, f_measure_M1 = calculate_metrics(predictions_M1, true_classes_binary)
        results[f'M1_threshold_{threshold}'] = (precision_M1, recall_M1, f_measure_M1)

        # Evaluate Model M2 (only at the default threshold)
        if threshold == 0.5:
            predictions_M2 = generate_predictions(probabilities_M2, threshold)
            precision_M2, recall_M2, f_measure_M2 = calculate_metrics(predictions_M2, true_classes_binary)
            results['M2_threshold_0.5'] = (precision_M2, recall_M2, f_measure_M2)
    
    return results

# Example usage (assuming probabilities_M1, probabilities_M2, and true_classes_binary are defined):
results = evaluate_models(probabilities_M1, probabilities_M2, true_classes_binary)

# Print the results
for model, metrics in results.items():
    print(f"{model} -> Precision: {metrics[0]:.4f}, Recall: {metrics[1]:.4f}, F-measure: {metrics[2]:.4f}")
