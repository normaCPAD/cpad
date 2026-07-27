from cpad.core.constraint import FunctionalDependency as FD
from cpad.modeling.closure import as_pairs, attribute_closure, candidate_keys, minimal_cover
from cpad.modeling.normalize import synthesize_3nf, decompose_bcnf
from cpad.modeling.report import _normal_form


def _pairs(fds):
    return as_pairs([FD(tuple(l), r) for l, r in fds])


def test_attribute_closure():
    fds = _pairs([(("A",), "B"), (("B",), "C")])
    assert attribute_closure({"A"}, fds) == {"A", "B", "C"}
    assert attribute_closure({"D"}, fds) == {"D"}


def test_candidate_keys():
    # A->B, B->C over {A,B,C,D}: D is non-determined, so the only key is {A, D}
    fds = _pairs([(("A",), "B"), (("B",), "C")])
    keys = candidate_keys({"A", "B", "C", "D"}, fds)
    assert frozenset({"A", "D"}) in keys


def test_minimal_cover_removes_redundancy():
    # A->C is implied by A->B, B->C and must be dropped
    fds = _pairs([(("A",), "B"), (("B",), "C"), (("A",), "C")])
    cover = minimal_cover(fds)
    assert (frozenset({"A"}), "C") not in cover
    assert len(cover) == 2


def test_3nf_not_bcnf_detection_and_decomposition():
    # R(A,B,C): AB->C, C->B  =>  3NF (B prime) but not BCNF (C not superkey)
    fds = _pairs([(("A", "B"), "C"), (("C",), "B")])
    attrs = {"A", "B", "C"}
    cover = minimal_cover(fds)
    keys = candidate_keys(attrs, cover)
    assert _normal_form(attrs, cover, keys) == "3NF"
    bcnf = decompose_bcnf(attrs, fds)
    # decomposition is lossless and every relation is in BCNF (split on C->B)
    assert any({"B", "C"} == set(r.attributes) for r in bcnf)


def test_3nf_synthesis_covers_attributes():
    fds = _pairs([(("A",), "B"), (("B",), "C")])
    rels = synthesize_3nf({"A", "B", "C", "D"}, fds)
    covered = set().union(*[r.attributes for r in rels])
    assert {"A", "B", "C", "D"} <= covered
