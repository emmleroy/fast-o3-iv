import pytest

from fast_o3_iv.preprocess.process_chem_data import ProcessChemData


@pytest.mark.parametrize(
    "target_source_id,input_type,experiment_id,var_original,var_output,year_range",
    [
        (
            "CESM2",
            "ghg_concentrations",
            "historical",
            "mole_fraction_of_methane_in_air",
            "mole_fraction_of_methane_in_air",
            range(1975, 2015),
        ),
        (
            "CESM2",
            "emissions",
            "ssp245",
            "CO_em_openburning",
            "CO_em_biomassburning",
            range(2015, 2055),
        ),
        (
            "CESM2-WACCM6",
            "emissions",
            "historical",
            "NOx_em_anthro",
            "NOx_em_anthro",
            range(1975, 2015),
        ),
        (
            "CESM2-WACCM6",
            "ghg_concentrations",
            "ssp245",
            "mole_fraction_of_methane_in_air",
            "mole_fraction_of_methane_in_air",
            range(2015, 2055),
        )
    ],
)
def test_process_chem_data(
    capsys,
    target_source_id,
    input_type,
    experiment_id,
    var_original,
    var_output,
    year_range,
):
    pipe = ProcessChemData(
        target_source_id=target_source_id,
        input_type=input_type,
        experiment_id=experiment_id,
        var_original=var_original,
        var_output=var_output,
        year_range=year_range,
    )

    ds = pipe.compute()
    ds.close()

    captured = capsys.readouterr().out
    institution = pipe.institution_id

    expected_suffix = (
        f"{institution}/{target_source_id}/{experiment_id}/inputs/{var_output}/"
        f"{institution}.{target_source_id}.{experiment_id}.inputs.{var_output}."
        f"{year_range[0]}-{year_range[-1]}.nc"
    )

    assert "Ready to save final concatenated file -->" in captured
    assert expected_suffix in captured
