import argparse

from deepinfer.utils.misc import get_model, get_dataset
from deepinfer.core import infer_data_precondition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['GC', 'BM', 'HP', 'PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='Test/validation dataset', required=True,
                        choices=['Bank Customer', 'German Credit', 'House Price', 'PIMA diabetes'])
    args = parser.parse_args()

    model = get_model(model=args.model, version=args.version)
    dataset = get_dataset(dataset=args.dataset, split='val')

    infer_data_precondition(model, dataset)
