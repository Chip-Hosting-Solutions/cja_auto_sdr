from scripts.update_test_counts import parse_counts


def test_parse_counts_with_paths():
    output = """
<Module tests/test_alpha.py>
  <Function test_one>
  <Function test_two>
<Module tests/unit/test_beta.py>
  <Function test_three>
<Module scripts/helper.py>
  <Function helper_func>
""".strip()
    counts = parse_counts(output)
    assert counts == {
        "test_alpha.py": 2,
        "test_beta.py": 1,
    }


def test_parse_counts_without_paths():
    output = """
<Module test_gamma.py>
  <Function test_a>
  <Function test_b>
""".strip()
    counts = parse_counts(output)
    assert counts == {"test_gamma.py": 2}
