from layers import Sequential, Linear


def create_mlp(layers, act_fun):
    modules = []
    for i, o in zip(layers[:-1], layers[1:]):
        modules.append(Linear(i, o))
        modules.append(act_fun())
    modules.pop(-1)
    model = Sequential(*modules)
    return model
