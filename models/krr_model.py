from sklearn.kernel_ridge import KernelRidge

def get_krr_model(params):
    return KernelRidge(
        alpha=params.get('alpha', 1.0),
        kernel=params.get('kernel', 'rbf')
    )