from opencompass.opencompass.utils import text_postprocessors as undertest


def test_last_number_postprocess():
    s = "this string 123 contains some 13123 numbers 12312312 find the -1.75 last example 1337.0"
    assert undertest.last_number_postprocess(s) == 1337.0
