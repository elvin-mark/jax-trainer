from layers import BasicResidualBlock1d, BasicResidualBlock2d, BatchNorm1d, BatchNorm2d, Conv1d, Conv2d, Flatten, Linear, MaxPool1d, MaxPool2d, ReLU, Sequential


def create_conv1d_classifier(image_shape=1000, input_channels=1, num_classes=10):
    modules = []
    i = 4
    while image_shape > 1:
        output_channels = 2**i
        modules.append(Sequential(
            Conv1d(input_channels, output_channels, bias=False, padding=1),
            BatchNorm1d(output_channels),
            ReLU(),
            MaxPool1d()
        ))
        input_channels = output_channels
        i += 1
        image_shape = image_shape // 2
    modules.append(Flatten((-1, image_shape * image_shape * output_channels)))
    modules.append(Linear(image_shape * image_shape *
                   output_channels, num_classes))
    return Sequential(*modules)


def create_residual1d_classifier(image_shape=28, input_channels=1, num_classes=10):
    modules = []
    i = 4
    while image_shape > 1:
        output_channels = 2**i
        modules.append(Sequential(
            BasicResidualBlock1d(input_channels, output_channels),
            MaxPool1d())
        )
        input_channels = output_channels
        i += 1
        image_shape = image_shape // 2
    modules.append(Flatten((-1, image_shape * image_shape * output_channels)))
    modules.append(Linear(image_shape * image_shape *
                   output_channels, num_classes))
    return Sequential(*modules)


def create_conv2d_classifier(image_shape=28, input_channels=1, num_classes=10):
    modules = []
    i = 4
    while image_shape > 1:
        output_channels = 2**i
        modules.append(Sequential(
            Conv2d(input_channels, output_channels, bias=False, padding=1),
            BatchNorm2d(output_channels),
            ReLU(),
            MaxPool2d()
        ))
        input_channels = output_channels
        i += 1
        image_shape = image_shape // 2
    modules.append(Flatten((-1, image_shape * image_shape * output_channels)))
    modules.append(Linear(image_shape * image_shape *
                   output_channels, num_classes))
    return Sequential(*modules)


def create_residual2d_classifier(image_shape=28, input_channels=1, num_classes=10):
    modules = []
    i = 4
    while image_shape > 1:
        output_channels = 2**i
        modules.append(Sequential(
            BasicResidualBlock2d(input_channels, output_channels),
            MaxPool2d())
        )
        input_channels = output_channels
        i += 1
        image_shape = image_shape // 2
    modules.append(Flatten((-1, image_shape * image_shape * output_channels)))
    modules.append(Linear(image_shape * image_shape *
                   output_channels, num_classes))
    return Sequential(*modules)
