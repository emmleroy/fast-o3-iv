Getting started
===============


### Step 0: Clone the repo

```bash
git clone https://github.com/emmleroy/fast-o3-iv.git
cd fast-o3-iv
```

### Step 1: Install `fast-o3-iv`

> **Note:** You can install with either **Makefile commands** or **manual commands**.  
> Pick one path and follow it end-to-end.

#### Option A: Makefile (recommended)

```bash
make create_environment
conda activate fast-o3-iv
make requirements
make test
```

#### Option B: Manual commands

Create environment:
```bash
conda create --name fast-o3-iv python=3.11 -y
conda install -n fast-o3-iv -c conda-forge poetry esmf esmpy xesmf -y
conda activate fast-o3-iv
```
Install dependencies:
```bash
poetry config virtualenvs.create false
poetry install
```
Run tests:
```bash
poetry run pytest --forked tests/
```

### Step 2. Download raw climate model output

--> fast_o3_iv/ingestion

| source_id      | institution_id | description          | URL                                     |
| ---------------| ---------------| ---------------------| --------------------------------------- |
| `CESM2`        | NCAR           | CESM2 Large Ensemble | https://gdex.ucar.edu/datasets/d651056/ | 
| `CESM2-WACCM6` | NCAR           |                      |                                         | 
                                                                                

Model output in data/external must follow the following directory structure:
```
└── data
     └── external       
          └── <institution_id> (NCAR)
                  └── <source_id> (CESM2)
                            └── <experiment_id> (historical)
                                    └── <variant_label> (LE2-1001.001)
                                              └── <variable_id> (TS)
```

### Step 3. Pre-process climate model output (data/external --> data/interim)

--> fast_o3_iv/preprocess

Pre-processing includes:
1. Concatenate monthly netCDF files across time for the requested years (i.e. 1975-2014)
2. Rename variables to CMIP6 convention is not already done
3. Normalize monthly timestamps to month-start
4. Write a single netCDF output to data/interim

Pre-processed model output in data/interim follows the following directory structure:
```
└── data
     └── interim       
          └── <institution_id> (NCAR)
                  └── <source_id> (CESM2)
                            └── <experiment_id> (historical)
                                    └── <variant_label> (LE2-1001.001)
                                              └── <variable_id> (ts)
```
Runnings tests:
Running a forked test isolates backend stacks per process
so that we can use two different engines (h5netcdf for CESM2 and netcdf4 for CESM2-WACCM6)

```
poetry run pytest --forked tests/ 
```
or just:
```
make test
```

### Step 4. Create features (data/interim --> data/interim)
--> fast_o3_iv/features

There's two main features we create from the interim data:
1) Applying a blurring kernel to the emission fields (features/emissions_filter.py)
2) Apply PCA regression to the 500 mb geopotential height fields (features/PC_reduction.py)

### Step 5. Create dataframes for testing/training (data/interim --> data/processed)
