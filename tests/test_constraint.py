from cpad.core.constraint import FunctionalDependency, DenialConstraint, Predicate, EQ, NE


def test_fd_str_simple_and_composite():
    assert str(FunctionalDependency(("A",), "B")) == "A -> B"
    assert str(FunctionalDependency(("A", "B"), "C")) == "(A, B) -> C"


def test_fd_to_denial_constraint():
    dc = FunctionalDependency(("state", "marital_status"), "single_exemp").to_dc()
    assert isinstance(dc, DenialConstraint)
    ops = {(p.left, p.op, p.right) for p in dc.predicates}
    assert ("state", EQ, "state") in ops
    assert ("marital_status", EQ, "marital_status") in ops
    assert ("single_exemp", NE, "single_exemp") in ops


def test_fd_flags():
    assert FunctionalDependency(("A", "B"), "C").is_composite
    assert not FunctionalDependency(("A",), "C").is_composite
    assert FunctionalDependency(("A", "B"), "C").arity == 2
