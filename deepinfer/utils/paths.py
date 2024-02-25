from pathlib import Path

module_path = Path(__file__).parent.parent.absolute()

datasets_path = module_path.parent / 'datasets'
models_path = module_path.parent / 'models'
results_path = module_path.parent / 'results'

results_path.mkdir(exist_ok=True)
