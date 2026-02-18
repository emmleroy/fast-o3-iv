import pytest

from fast_o3_iv.preprocess.process_met_data import ProcessMetData


@pytest.mark.parametrize(
    "model,scenario,var_original,member,years, level",
    [
        ("CESM2", "historical", "TREFHT", "LE2-1001.001", range(1975, 2015), None),
        ("CESM2", "ssp245", "U", "001", range(2034, 2036), None),
        ("CESM2-WACCM6", "historical", "Z3", "wa6_ic1.004", range(1988, 1999), 500),
        ("CESM2-WACCM6", "ssp245", "PSL", "007", range(2044, 2048), None),
    ],
)
def test_process_met_data(capsys, model, scenario, var_original, member, years, level):
    pipe = ProcessMetData(
        source_id=model,
        experiment_id=scenario,
        variant_label=member,
        var_original=var_original,
        year_range=years,
        level=level,
    )

    ds = pipe._compute()
    ds.close()

    captured = capsys.readouterr().out
    institution = pipe.institution_id
    output_var = pipe.var_output

    expected_suffix = (
        f"{institution}/{model}/{scenario}/{member}/{output_var}/"
        f"{institution}.{model}.{scenario}.{member}.{output_var}.{years[0]}-{years[-1]}.nc"
    )

    assert "Ready to save final concatenated file -->" in captured
    assert expected_suffix in captured
