# fast-o3-iv

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fast surrogate modeling of ozone internal variability

## Project Organization

```
├── LICENSE                  
├── Makefile                 
├── README.md                
├── scripts/            
├── data/
│   ├── external/            <- Raw, immutable data (climate model output from cmip6 or cesm)
│   ├── interim/             <- Pre-processed climate model output (netcdf4 files)
│   └── processed/           <- The final, canonical data sets for modeling (parquet files)
│       ├── model_inputs/
|       └── model_outputs/
├── models/                  <- Path to trained models
├── pyproject.toml           <- Project configuration file
├── figures                  <- Generated figures
│   └── JAMES/         
├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
│                               generated with `pip freeze > requirements.txt`
├── tests/                  
└── src/fast_o3_iv/          <- Source code for use in this project.
    ├── __init__.py          
    ├── config.py            <- Store useful variables and configuration
    ├── metrics.py           <- Helper funcs to calculate performance metrics
    ├── utils.py             <- Helpful utlity functions
    ├── preprocess/          <- Code to pre-process model output (data/external -> data/internal) 
    ├── features/            <- Code to create features (data/interim --> data/interim)
    ├── tabularize/          <- Code to create dataframes for modeling (data/interim --> data/processed)     
    └── modeling/               
        ├── __init__.py 
        ├── predict.py       <- Code to run model inference with trained models          
        └── train.py         <- Code to train models
    
```

--------

