from surprise import accuracy
from sklearn.metrics import precision_score, recall_score
import numpy as np

def compute_rmse(predictions):
    return accuracy.rmse(predictions, verbose=False)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = {}
    for uid, iid, true_r, est, _ in predictions:
        user_est_true.setdefault(uid, []).append((est, true_r))
    
    precisions = []
    recalls = []
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        
        true_positives = sum((true_r >= threshold) for est, true_r in top_k)
        total_relevant = sum((true_r >= threshold) for est, true_r in user_ratings)
        
        precision = true_positives / k
        recall = true_positives / total_relevant if total_relevant != 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    
    return mean_precision, mean_recall
