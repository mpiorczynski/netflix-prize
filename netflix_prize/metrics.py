from collections import defaultdict


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, r_true, r_est, _ in predictions:
        user_est_true[uid].append((r_est, r_true))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        num_relevant = sum((r_true >= threshold) for (_, r_true) in user_ratings)

        # Number of recommended items in top k
        num_recommended_k = sum((r_est >= threshold) for (r_est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        num_relevant_and_reccomended_k = sum(
            ((r_true >= threshold) and (r_est >= threshold))
            for (r_est, r_true) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When num_recommended_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = num_relevant_and_reccomended_k / num_recommended_k if num_recommended_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When num_relevant is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = num_relevant_and_reccomended_k / num_relevant if num_relevant != 0 else 0

    precision = sum(p for p in precisions.values()) / len(precisions)
    recall = sum(r for r in recalls.values()) / len(recalls)
    return precision, recall
