import argparse

from deepinfer.utils.misc import get_model
from deepinfer.core import infer_data_precondition, match_features_to_precondition
from trustbench.utils.misc import get_datasets_configs, list_datasets, load_dataset

DATASETS_CONFIGS = get_datasets_configs()
dataset_choices = sorted(list(DATASETS_CONFIGS.keys()))
datasets = list_datasets()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['GC', 'BM', 'HP', 'PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='Test/validation dataset', required=True,
                        choices=dataset_choices)

    args = parser.parse_args()

    model = get_model(model=args.model, version=args.version)
    dataset = load_dataset(name=args.dataset, path=datasets[args.dataset], config=DATASETS_CONFIGS[args.dataset])

    wp = infer_data_precondition(model)
    wp_dict = match_features_to_precondition(wp, dataset.splits['val'].features)

    print(wp_dict)
