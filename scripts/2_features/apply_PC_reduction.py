import argparse

from fast_o3_iv.features.PC_reduction import ApplyPCReduction

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-s', '--scenario', type=str, required=True)
parser.add_argument("-v", "--var_input", required=True)
parser.add_argument("-t", "--training_member", required=True)

args = parser.parse_args()

model = args.model                  # CESM2-WACCM6
scenario = args.scenario            # historical
var_input = args.var_input          # [CO,NOx]_em_openburning, [CO,NOx]_em_anthro
training_member = args.training_member

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

pipe = ApplyPCReduction(
    input_source_id = model,
    experiment_id = scenario,
    training_member = training_member,
    var_input = var_input,
    year_range = year_range,
    k = 5 # default
)

pipe.fit()
for member in pipe.members:
    print(member)
    pipe.compute_and_save(member, save=True)
