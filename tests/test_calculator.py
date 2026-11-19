import pytest

from audit_env.env.calculator import CalcError, evaluate


@pytest.mark.parametrize("expr,value", [
    ("1+2*3", "7"),
    ("(1+2)*3", "9"),
    ("10/4", "2.5"),
    ("-3 + +5", "2"),
    ("1/3", "0.333333333333"),
    ("2/3", "0.666666666667"),
    ("123456.78 * 514.286", "63492093.55908"),
    (".5 + 1.25", "1.75"),
    ("0.1+0.2", "0.3"),
    ("--1", "1"),
    ("1-1", "0"),
])
def test_arithmetic(expr, value):
    assert evaluate(expr) == value


@pytest.mark.parametrize("expr", [
    "", " ", "1/0", "2**3", "__import__('os')", "1e5", "abc", "1+", "(1", "1)", "()", "1..2", ".",
    "os.system('x')", "1;2", "[1]", "1 2", "*1", "x" * 10, "1" * 41, "9" * 30 + "*" + "9" * 30,
    "(" * 25 + "1" + ")" * 25, "1+" * 150 + "1",
])
def test_rejects(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_non_string():
    with pytest.raises(CalcError):
        evaluate(12)  # type: ignore[arg-type]


def test_no_eval_in_source():
    import inspect

    import audit_env.env.calculator as calc
    src = inspect.getsource(calc)
    assert "eval(" not in src.replace("evaluate(", "")
    assert "exec(" not in src
