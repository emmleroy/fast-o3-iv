import pytest
from fast_o3_iv.preprocess.process_met_data import ProcessMetData
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG 


def test_process_met_data(capsys):
    """test that ProcessMetData.compute() works and returns expected suffix
    """

    model = "CESM2"
    scenario = "historical"
    var_original = "TS"
    member = "LE2-1001.001"
    years = range(1975, 2015)

    pipe = ProcessMetData(
        source_id=model,
        experiment_id=scenario,
        variant_label=member,
        var_original=var_original,
        year_range=years,
    )

    pipe.compute_and_save(save=False)

    captured = capsys.readouterr().out

    model_config = MODEL_CONFIG[model]
    institution = model_config['institution_id']

    variable_map = model_config['variable_map']
    output_var = variable_map[var_original]

    expected_suffix = (
        f"{institution}/{model}/{scenario}/{member}/{output_var}/"
        f"{institution}.{model}.{scenario}.{member}.{output_var}.{years[0]}-{years[-1]}.nc"
    )
    assert "Ready to save final concatenated file -->" in captured
    assert expected_suffix in captured

