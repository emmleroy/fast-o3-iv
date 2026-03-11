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

#pipe.compute()
#pipe.concat_saved()

idx = 1
inpaths = pipe._glob_all_filepaths()
inpath = inpaths[idx]
year_groups = [
                "2015-2024",
                "2025-2034",
                "2035-2044",
                "2045-2054"
            ]
ds = pipe._process_subset(inpath)
pipe._save_file_subsets(ds, year_groups[idx])