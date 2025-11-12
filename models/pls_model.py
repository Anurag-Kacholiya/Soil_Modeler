from sklearn.cross_decomposition import PLSRegression

def get_plsr_model(params):
    return PLSRegression(n_components=params.get('n_components', 10))