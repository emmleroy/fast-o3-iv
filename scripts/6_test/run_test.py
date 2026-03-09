import argparse

from fast_o3_iv.modeling.train import DatasetConfig
from fast_o3_iv.modeling.predict import PredictPipeline


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--model_key', type=str, required=True)
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-s', '--scenario', type=str, required=True)
parser.add_argument("-v", "--var_setname", required=True)

args = parser.parse_args()

model_key = args.model_key
model = args.model
scenario = args.scenario
var_setname = args.var_setname

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

train_config = DatasetConfig(
    source_id = model,
    cmip_experiment_id = scenario,
    variable_setname = var_setname, # local
    year_range = year_range
)

predict_config = train_config # validation/testing

pipe = PredictPipeline(
    model_key=model_key,
    train_config=train_config,
    predict_config=predict_config,
    is_test=False
    )

stats_pipeline = pipe.load_model()

for member in train_config.remaining_members:
    print(member, flush=True)
    pipe.predict_and_save_results(member, stats_pipeline, save=True)
