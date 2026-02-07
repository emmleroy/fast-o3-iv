Getting started
===============

### Step 1. Install fast-o3-iv 

TO DO: update this once package is minimally functional

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

### Step 3. Pre-process climate model output

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
