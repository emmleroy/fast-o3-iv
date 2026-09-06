"""run_process_met.py

Script to pre-process met variables (Z3, TS, U, V, etc..)

"""

import argparse

from fast_o3_iv.preprocess.process_met_data import ProcessMetData

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-s', '--scenario', type=str, required=True)
parser.add_argument("-l", "--variant_label", default=None)
parser.add_argument("-v", "--var_original", required=True)
parser.add_argument("-o", "--level", default=None)

args = parser.parse_args()

model = args.model                  # CESM2-WACCM6
scenario = args.scenario            # historical
variant_label = args.variant_label  # wa6_ic1.001
var_original = args.var_original    # U, V
level = args.level                  # surface, None, or numeric hPa

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

pipe = ProcessMetData(
    source_id = model,
    experiment_id = scenario,
    variant_label=variant_label,
    var_original = var_original,
    year_range = year_range,
    level=level
)

pipe.compute_and_save(save=True)
