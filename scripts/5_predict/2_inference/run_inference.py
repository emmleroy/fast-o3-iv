import argparse

from fast_o3_iv.modeling.train import DatasetConfig
from fast_o3_iv.modeling.predict import PredictPipeline


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train-source", required=True, help="Source ID for training data")
parser.add_argument("-p", "--predict-source", required=True, help="Source ID for prediction data")
parser.add_argument("-k", "--model-key", required=True, help="Key for the trained model to load")
parser.add_argument("-e", "--experiment", required=True, help="CMIP experiment ID, e.g. historical")
parser.add_argument("-v", "--variable-set", required=True, help="Variable set name")

args = parser.parse_args()

train_source = args.train_source
predict_source = args.predict_source
model_key = args.model_key
experiment = args.experiment
variable_set = args.variable_set

year_range = range(1975,2015) if args.experiment == "historical" else range(2015,2055)

train_config = DatasetConfig(
    source_id=train_source,
    cmip_experiment_id=experiment,
    variable_setname=variable_set,
    year_range=year_range,
)

predict_config = DatasetConfig(
    source_id=predict_source,      # different! inference on CESM2-LE instead of CESM2-WACCM6
    cmip_experiment_id=experiment, # same as train
    variable_setname=variable_set, # same as train
    year_range=year_range,         # same as train
)

pipe = PredictPipeline(
    model_key=model_key,
    train_config=train_config,
    predict_config=predict_config,
    is_test=False
    )

stats_pipeline = pipe.load_model()

for member in predict_config.members: # predict all members --> will not overwrite val and test results if already run
    print(member, flush=True)
    pipe.predict_and_save_results(member, stats_pipeline, save=True)
