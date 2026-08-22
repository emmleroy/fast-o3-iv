# Maps variable setnames -> feature and target variable names

VARIABLE_SET = {
  "setL" : { # local (prev setK)
    "target" : ["mda8o3"],
    "variables" : ['tas', 'pr', 'psl', 'rsds','huss', 'ua', 'va', 'NOx_em_anthro_dirblur', 'NOx_em_openburning_dirblur', 'CO_em_anthro_dirblur', 'CO_em_openburning_dirblur'],
    "input_variables" :  [
        'mole_fraction_of_methane_in_air'
    ],
    "pc_variables" : ["hemis_pcs_zg500_top5"],
    "time_variables" : ["month"]
  },
  "setG" : { # global (prev setG)
    "target" : ["mda8o3"],
    "variables" : ['tas', 'pr', 'psl', 'rsds','huss', 'ua', 'va', 'NOx_em_anthro_dirblur', 'NOx_em_openburning_dirblur', 'CO_em_anthro_dirblur', 'CO_em_openburning_dirblur'],
    "input_variables" :  [
        'mole_fraction_of_methane_in_air'
    ],
    "pc_variables" : ["hemis_pcs_zg500_top5"],
    "time_variables" : ["month"],
    "location_variables": ['lat', 'lon', 'orog', 'sftlf']
  },
}

pc_variable_sets = {
  "hemis_5pcs_zg500" : [
           'pc1_zg500',
           'pc2_zg500',
           'pc3_zg500',
           'pc4_zg500',
           'pc5_zg500',
  ]
}