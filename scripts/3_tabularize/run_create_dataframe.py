import argparse

from fast_o3_iv.tabularize.create_dataframe import CreateDataframe

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-s', '--scenario', type=str, required=True)
parser.add_argument("-v", "--var_setname", required=True)

args = parser.parse_args()

model = args.model                  # CESM2-WACCM6
scenario = args.scenario            # historical, ssp370
var_setname = args.var_setname      # setC

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

pipe = CreateDataframe(
    input_source_id = model,
    experiment_id = scenario,
    variable_setname = var_setname,
    year_range = year_range,
)

for member in pipe.members:
    print(member)
    df = pipe.compute_and_save(member, save=True)