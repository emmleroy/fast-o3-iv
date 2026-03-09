import argparse

from fast_o3_iv.modeling.train import TrainPipeline, DatasetConfig


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--model_key', type=str, required=True)
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-s', '--scenario', type=str, required=True)
parser.add_argument("-v", "--var_setname", required=True)
parser.add_argument("-t", "--training_member", required=True)

args = parser.parse_args()

model_key = args.model_key
model = args.model
scenario = args.scenario
var_setname = args.var_setname
training_member = args.training_member

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

config = DatasetConfig(
    source_id = model,
    cmip_experiment_id = scenario,
    variable_setname = var_setname, # local
    year_range = year_range
)

pipe = TrainPipeline(model_key=model_key, config=config, is_test=False)
pipe.train_and_save_model(save=True)
