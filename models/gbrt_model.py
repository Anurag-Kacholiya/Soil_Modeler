from sklearn.ensemble import GradientBoostingRegressor

def get_gbrt_model(params):
    return GradientBoostingRegressor(
        learning_rate=params.get('learning_rate', 0.1),
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 3)
    )