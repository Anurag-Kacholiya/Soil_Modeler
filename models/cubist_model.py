from cubist import Cubist

def get_cubist_model(params):
    return Cubist(
        n_rules=params.get('n_rules',2)
    )