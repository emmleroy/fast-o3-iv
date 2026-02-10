import pytest

from fast_o3_iv.preprocess.process_met_data import ProcessMetData


def test_process_met_data(capsys):
    """Test that ProcessMetData.compute() works and returns expected suffix
    """
    model = "CESM2"
    scenario = "ssp245"
    var_original = "TREFHT"
    member = "001"
    years = range(2015, 2055)

    pipe = ProcessMetData(
        source_id=model,
        experiment_id=scenario,
        variant_label=member,
        var_original=var_original,
        year_range=years,
    )

    ds = pipe.compute()
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

