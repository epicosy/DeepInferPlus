import sys
import json
import argparse
import pandas as pd

from pathlib import Path

from deepinfer.utils.misc import get_model
from deepinfer.utils.paths import results_path
from deepinfer.core import compute_threshold, check_prediction, PREDICTION_INTERVALS, CONDITIONS


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
        val_features = pd.read_csv(args.val_features)
        threshold, wp_dict = compute_threshold(model, val_features, prediction_interval=args.prediction_interval,
                                               condition=args.condition)
        wp_dict = {k: float(v) for k, v in wp_dict.items()}
        analysis = {'threshold': float(threshold), 'wp_dict': wp_dict}

        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=4)

    elif args.action == 'infer':
        test_features = pd.read_csv(args.test_features)

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
