import argparse

from fast_o3_iv.preprocess.calc_mda8 import ProcessMDA8O3

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument("-s", "--scenario", default=None)
parser.add_argument("-v", "--var_original", required=True)
parser.add_argument("-l", "--member", default=None)

args = parser.parse_args()

source_id = args.model
experiment_id = args.scenario
member = args.member
var_original = args.var_original

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

pipe = ProcessMDA8O3(
    source_id=source_id,
    experiment_id=experiment_id,
    variant_label=member,
    var_original=var_original,
    year_range=year_range
)

pipe.compute()
pipe.concat_saved()