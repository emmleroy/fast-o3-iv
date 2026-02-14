import argparse

from fast_o3_iv.features.emissions_filter import ApplyEmissionsFilter

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-s', '--scenario', type=str, required=True)
parser.add_argument("-v", "--var_input", required=True,
                    choices=["CO_em_anthro", "CO_em_openburning", "NOx_em_anthro", "NOx_em_openburning"])
parser.add_argument("-o", "--method", required=True, choices=["isotropic", "downwind"])
parser.add_argument("-l", "--variant_label", default=None)

args = parser.parse_args()

model = args.model                  # CESM2-WACCM6
scenario = args.scenario            # 'historical'
var_input = args.var_input          # 'CO_em_openburning' #CO_em_anthro, NOx_em_anthro, NOx_em_openburning, 
method = args.method                #'downwind'
variant_label = args.variant_label  # 'wa6_ic1.001'

if args.method == "downwind" and not args.variant_label:
    parser.error("--variant_label is required when --method=downwind")

year_range = range(1975,2015) if args.scenario == "historical" else range(2015,2055)

pipe = ApplyEmissionsFilter(
    source_id = model,
    experiment_id = scenario,
    var_input = var_input,
    year_range = year_range,
    method = method,
    variant_label=variant_label
)

pipe.compute_and_save(save=True)