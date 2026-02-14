import pytest

from fast_o3_iv.tabularize.create_dataframe import CreateDataframe

def test_create_dataframes(capsys):
    """Test that ProcessMetData.compute() works and returns expected suffix
    """

    model = "CESM2-WACCM6"
    scenario = "historical"
    var_setname = "setA"
    member = "wa6_ic2.003"
    years = range(1975, 2015)

    pipe = CreateDataframe(
        input_source_id = model,
        experiment_id = scenario,
        variable_setname = var_setname,
        year_range = years,
    )

    df = pipe._compute(member)

    captured = capsys.readouterr().out

    institution = pipe.institution_id

    expected_suffix = (
        f"{institution}/{model}/{scenario}/{member}/{var_setname}/"
        f"{institution}.{model}.{scenario}.{member}.{var_setname}.model_inputs.{years[0]}-{years[-1]}.parquet"
    )
    assert "Ready to save df -->" in captured
    assert expected_suffix in captured
