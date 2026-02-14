# Maps variable setnames -> feature and target variable names
# met variables must be first
VARIABLE_SET = {
  "setA" : {
    "target" : ["mda8o3"],
    "met" : ['tas', 'pr', 'psl', 'rsds','huss', 'ua', 'va'],
    "chem" : ['NOx_em_anthro_dwblur', 'NOx_em_openburning_dwblur', 'CO_em_anthro_dwblur', 'CO_em_openburning_dwblur', 'mole_fraction_of_methane_in_air'],
    "pc_variables" : ["hemis_5pcs_zg500"],
    "time_variables" : ["month"],
  },
  "setB" : {
    "target" : ["mda8o3"],
    "met" : ['tas', 'pr', 'psl', 'rsds','huss', 'ua', 'va'],
    "chem" : ['NOx_em_anthro_dwblur', 'NOx_em_openburning_dwblur', 'CO_em_anthro_dwblur', 'CO_em_openburning_dwblur', 'mole_fraction_of_methane_in_air'],
    "pc_variables" : ["hemis_5pcs_zg500"],
    "time_variables" : ["month"],
    "location_variables": ['lat', 'lon', 'orog', 'sftlf'],
  },
  "setC" : {
    "target" : ["mda8o3"],
    "met" : ['tas', 'pr', 'psl', 'rsds','huss', 'ua', 'va'],
    "chem" : ['NOx_em_anthro_blur', 'NOx_em_openburning_blur', 'CO_em_anthro_blur', 'CO_em_openburning_blur', 'mole_fraction_of_methane_in_air'],
    "pc_variables" : ["hemis_5pcs_zg500"],
    "time_variables" : ["month"],
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