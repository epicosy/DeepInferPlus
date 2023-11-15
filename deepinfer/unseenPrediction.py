import time
import pandas as pd
import argparse

from deepinfer.utils.misc import get_model, get_dataset, UnseenSplit
from deepinfer.core import unseen_prediction

# TODO: check if necessary to remove the warning
pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implying trustworthiness on the model’s prediction using inferred '
                                                 'data preconditions')
    parser.add_argument('--model', type=str, help='Model to infer the precondition', required=True,
                        choices=['GC', 'BM', 'HP', 'PD'])
    parser.add_argument('--version', type=int, help='Version of the model', required=True)
    parser.add_argument('--dataset', type=str, help='Unseen validation dataset', required=True,
                        choices=['Bank Customer', 'German Credit', 'House Price', 'PIMA diabetes'])
    args = parser.parse_args()

    model = get_model(model=args.model, version=args.version)
    dataset = get_dataset(dataset=args.dataset, split=UnseenSplit())

    time_start = time.time()
    unseen_prediction(model, dataset)
    elapsed_time = time.time() - time_start

    print(f"Elapsed Time: {elapsed_time}")
