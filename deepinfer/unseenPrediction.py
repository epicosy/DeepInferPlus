import time
import pandas as pd
import argparse

from deepinfer.utils.misc import get_model
from deepinfer.core import compute_threshold, check_prediction

from trustbench.utils.misc import get_datasets_configs, list_datasets, load_dataset

DATASETS_CONFIGS = get_datasets_configs()
dataset_choices = sorted(list(DATASETS_CONFIGS.keys()))
datasets = list_datasets()

# TODO: check if necessary to remove the warning
pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implying trustworthiness on the model’s prediction using inferred '
                                                 'data preconditions')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['GC', 'BM', 'HP', 'PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='Unseen validation dataset', required=True,
                        choices=dataset_choices)
    args = parser.parse_args()

    model = get_model(model=args.model, version=args.version)
    dataset = load_dataset(name=args.dataset, path=datasets[args.dataset], config=DATASETS_CONFIGS[args.dataset])

    time_start = time.time()
    threshold, wp_dict = compute_threshold(model, dataset.splits['val'].features)
    results = check_prediction(model, features=dataset.splits['test'].features, labels=dataset.splits['test'].labels,
                               threshold=threshold, wp_dict=wp_dict)
    elapsed_time = time.time() - time_start
    print(results)
    pd.DataFrame([results]).to_csv(f'./results/{args.model}{args.version}.csv', index=False)

    print(f"Elapsed Time: {elapsed_time}")
