from sklearn.svm import SVR

def get_svr_model(params):
    return SVR(
        C=params.get('C', 1.0),
        epsilon=params.get('epsilon', 0.1),
        kernel=params.get('kernel', 'rbf')
    )