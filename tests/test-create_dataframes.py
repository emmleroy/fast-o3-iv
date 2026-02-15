import pytest

from fast_o3_iv.tabularize.create_dataframe import CreateDataframe


@pytest.mark.parametrize(
    "model,scenario,var_setname,member,years",
    [
        ("CESM2", "historical", "setA", "LE2-1181.010", range(1975, 2015)),
        ("CESM2-WACCM6", "historical", "setA", "wa6_ic2.003", range(1975, 2015)),
    ],
)
def test_create_dataframes(capsys, model, scenario, var_setname, member, years):
    pipe = CreateDataframe(
        input_source_id=model,
        experiment_id=scenario,
        variable_setname=var_setname,
        year_range=years,
    )

    _ = pipe._compute(member)
    captured = capsys.readouterr().out
    institution = pipe.institution_id

    expected_suffix = (
        f"{institution}/{model}/{scenario}/{member}/{var_setname}/"
        f"{institution}.{model}.{scenario}.{member}.{var_setname}.model_inputs."
        f"{years[0]}-{years[-1]}.parquet"
    )

    assert "Ready to save df -->" in captured
    assert expected_suffix in captured
