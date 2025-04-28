import pandas as pd

def load_data(data_path="data/u.data", item_path="data/u.item", user_path="data/u.user"):
    # Load ratings data
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_path, sep='\t', names=ratings_cols, encoding='latin-1')
    
    # Load movies data
    item_cols = ['item_id', 'title'] + [str(i) for i in range(22)]
    items = pd.read_csv(item_path, sep='|', names=item_cols, usecols=[0, 1], encoding='latin-1')
    
    # Load users data (optional)
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(user_path, sep='|', names=user_cols, encoding='latin-1')
    
    return ratings, items, users

def preprocess_data(ratings):
    # Example preprocessing: drop timestamp
    ratings = ratings.drop(columns=['timestamp'])
    return ratings
