import pytest

from fast_o3_iv.preprocess.process_chem_data import ProcessChemData

def test_process_chem_data(capsys):
    """Test that ProcessMetData.compute() works and returns expected suffix
    """
    target_source_id = 'CESM2'
    input_type = 'ghg_concentrations'
    experiment_id = "ssp245"
    var_original = "mole_fraction_of_methane_in_air" # CO_em_anthro for CESM2, emiss_anthro for CESM2-WACCM6
    var_output = "mole_fraction_of_methane_in_air"
    year_range  = range(2015, 2055)

    pipe = ProcessChemData(
        target_source_id=target_source_id,
        input_type=input_type,
        experiment_id=experiment_id,
        var_original=var_original,
        var_output=var_output,
        year_range=year_range
    )

    ds = pipe.compute()
    ds.close()
    
    captured = capsys.readouterr().out

    institution = pipe.institution_id
    output_var = pipe.var_output

    expected_suffix = (
        f"{institution}/{target_source_id}/{experiment_id}/inputs/{var_output}/"
        f"{institution}.{target_source_id}.{experiment_id}.inputs.{var_output}.{year_range[0]}-{year_range[-1]}.nc"
    )
    assert "Ready to save final concatenated file -->" in captured
    assert expected_suffix in captured

