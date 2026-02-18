import argparse

from fast_o3_iv.preprocess.process_chem_data import ProcessChemData

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument("-s", "--scenario", default=None)
parser.add_argument('-t', '--input_type', type=str, required=True)
parser.add_argument("-v", "--var_original", required=True)
parser.add_argument("-o", "--var_output", default=None)

args = parser.parse_args()

target_source_id = args.model
experiment_id = args.scenario
input_type = args.input_type
var_original = args.var_original
var_output = args.var_output

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

pipe = ProcessChemData(
    target_source_id = target_source_id,
    input_type = input_type,
    experiment_id = experiment_id,
    var_original = var_original,
    var_output = var_output,
    year_range = year_range
)

pipe.compute_and_save(save=True)
