import sys
import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from deepinfer.utils.misc import get_model
from deepinfer.utils.paths import results_path
from deepinfer.core import compute_threshold, check_prediction, PREDICTION_INTERVALS, CONDITIONS, get_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Data Precondition')
    parser.add_argument('-m', '--model', type=str, help='Path to the model', required=True)
    parser.add_argument('-wd', '--workdir', type=str, help='Working directory', required=False)
    parser.add_argument('-c', '--condition', type=str, help='Condition to check', choices=CONDITIONS,
                        default=CONDITIONS[0])

    action_parser = parser.add_subparsers(dest='action')

    analyze_parser = action_parser.add_parser('analyze')
    analyze_parser.add_argument('-vx', '--val_features', type=str, help='Validation features',
                                required=True)
    analyze_parser.add_argument('-pi', '--prediction_interval', type=float, help='Prediction intervals',
                                choices=PREDICTION_INTERVALS, default=PREDICTION_INTERVALS[2])

    infer_parser = action_parser.add_parser('infer')
    infer_parser.add_argument('-tx', '--test_features', type=str, help='Test features', required=True)

    args = parser.parse_args()
    model = get_model(model_path=args.model)
    working_dir = Path(args.workdir) if args.workdir else results_path
    analysis_path = working_dir / 'analysis.json'
    implications_path = working_dir / 'implications.csv'
    violations_path = working_dir / 'violations.csv'
    satisfactions_path = working_dir / 'satisfactions.csv'

    if args.action == 'analyze':
        val_path = Path(args.val_features)

        if not val_path.exists():
            print(f"Could not find validation features file {val_path}", file=sys.stderr)
            exit()

        if val_path.suffix == '.npy':
            val_features = np.load(val_path)

            if len(val_features.shape) > 2:
                val_features = get_features(model, val_features, output_path=working_dir / 'val_features.csv')

        elif val_path.suffix == '.csv':
            # TODO: add case for files with no headers
            val_features = pd.read_csv(val_path, delimiter=',')
        else:
            print(f"Unsupported file format {val_path.suffix}", file=sys.stderr)
            exit()

        threshold, wp_dict = compute_threshold(model, val_features, prediction_interval=args.prediction_interval,
                                               condition=args.condition)

        wp_dict = {k: float(v) for k, v in wp_dict.items()}
        analysis = {'threshold': float(threshold), 'wp_dict': wp_dict}

        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=4)

    elif args.action == 'infer':
        test_path = Path(args.test_features)

        if not test_path.exists():
            print(f"Could not find test features file {test_path}", file=sys.stderr)
            exit()

        if test_path.suffix == '.npy':
            test_features = np.load(test_path)

            if len(test_features.shape) > 2:
                test_features = get_features(model, test_features, output_path=working_dir / 'test_features.csv')

        elif test_path.suffix == '.csv':
            test_features = pd.read_csv(test_path, delimiter=',')
        else:
            print(f"Unsupported file format {test_path.suffix}", file=sys.stderr)
            exit()

        with analysis_path.open(mode='r') as af:
            analysis = json.load(af)

        implications, violations, satisfactions = check_prediction(features=test_features,
                                                                   threshold=analysis['threshold'],
                                                                   wp_dict=analysis['wp_dict'],
                                                                   condition=args.condition)

        implications.to_csv(implications_path, index=False)
        violations.to_csv(violations_path)
        satisfactions.to_csv(satisfactions_path)

    else:
        print("Please specify a command ['analyze', 'infer'].", file=sys.stderr)
        exit()
