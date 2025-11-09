import pytest
from bitformat import Expression


def test_expression_creation():
    a = Expression(" {3 }")
    b = Expression("{3 + a}")
    c = Expression("{3 + a + b}")
    d = Expression("{x * (3 + b // 4)}")

    assert a.evaluate() == 3
    assert a.evaluate(penguin= 54.3) == 3
    assert b.evaluate(a=4) == 7
    assert c.evaluate(a=4, b=5) == 12
    assert d.evaluate(x=4, a=4, b=5) == 16


def test_disallowed_expressions():
    for bad_value in ["{a(4)}", "", "{}", "{{}", "{__a}", "{1 + f__d}", "{a=4;}"]:
        with pytest.raises(ValueError):
            _ = Expression(bad_value)
