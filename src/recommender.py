from surprise import Dataset, Reader, KNNBasic
from surprise import SVD
from surprise.model_selection import train_test_split

def build_trainset(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    return trainset

def user_based_cf(trainset, k=20, sim_options=None):
    if sim_options is None:
        sim_options = {'name': 'cosine', 'user_based': True}
    
    algo = KNNBasic(k=k, sim_options=sim_options)
    algo.fit(trainset)
    return algo

def item_based_cf(trainset, k=20, sim_options=None):
    if sim_options is None:
        sim_options = {'name': 'cosine', 'user_based': False}
    
    algo = KNNBasic(k=k, sim_options=sim_options)
    algo.fit(trainset)
    return algo

def get_top_n(predictions, n=10):
    from collections import defaultdict
    
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n
