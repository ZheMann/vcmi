from constrained_net.data.data_factory import DataFactory
from constrained_net.constrained_net import ConstrainedNet
import argparse

parser = argparse.ArgumentParser(
    description='Train the constrained_net',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fc_layers', type=int, default=2, required=False, help='Number of fully-connected layers [default: 2].')
parser.add_argument('--fc_size', type=int, default=1024, required=False, help='Number of neurons in Fully Connected layers')
parser.add_argument('--epochs', type=int, required=False, default=1, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size')
parser.add_argument('--model_name', type=str, required=False, help='Name for the model')
parser.add_argument('--model_path', type=str, required=False, help='Path to model to continue training (*.h5)')
parser.add_argument('--dataset', type=str, required=False, help='Path to dataset to train the constrained_net')
parser.add_argument('--height', type=int, required=False, default=480, help='Input Height [default: 480]')
parser.add_argument('--width', type=int, required=False, default=800, help='Width of CNN input dimension [default: 800]')
parser.add_argument('--constrained', type=int, required=False, default=0, help='Include constrained layer')

def run_locally():
    fc_size = 1024
    fc_layers = 2
    n_epochs = 1
    cnn_height = 480
    cnn_width = 800
    batch_size = 32
    use_constrained_layer = True
    model_path = None
    model_name = None
    dataset_path = "C:\\Git\\video-based-device-identification\\main\\datasets\\iPhone"

    return fc_size, fc_layers, n_epochs, cnn_height, cnn_width, batch_size, use_constrained_layer, model_path, model_name, dataset_path

if __name__ == "__main__":
    DEBUG = True

    if DEBUG:
        fc_size, fc_layers, n_epochs, cnn_height, cnn_width, batch_size, use_constrained_layer, model_path, model_name, dataset_path = run_locally()
    else:
        args = parser.parse_args()
        fc_size = args.fc_size
        fc_layers = args.fc_layers
        n_epochs = args.epochs
        cnn_height = args.height
        cnn_width = args.width
        batch_size = args.batch_size
        use_constrained_layer = args.constrained == 1
        model_path = args.model_path
        model_name = args.model_name
        dataset_path = args.dataset

    data_factory = DataFactory(input_dir=dataset_path,
                               batch_size=batch_size,
                               height=cnn_height,
                               width=cnn_width)

    num_classes = len(data_factory.get_class_names())
    train_ds = data_factory.get_tf_train_data()
    filename_ds, test_ds = data_factory.get_tf_test_data()

    constr_net = ConstrainedNet(constrained_net=use_constrained_layer)
    if model_path:
        constr_net.set_model(model_path)
    else:
        # Create new model
        constr_net.create_model(num_classes, fc_layers, fc_size, cnn_height, cnn_width, model_name)

    constr_net.print_model_summary()
    constr_net.train(train_ds=train_ds, val_ds=test_ds, epochs=n_epochs)
