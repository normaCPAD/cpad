"""Command-line interface.

    cpad analyze data.csv [--learner routed|discrete|gated] [--cap 300]
                           [--top 10] [--json model.json]

Given a tabular file, learns the functional dependencies / denial constraints that
govern it and prints a data model: constraints, candidate keys, current normal form,
a proposed 3NF/BCNF decomposition, and the worst constraint-violating cells.
"""
from __future__ import annotations
import argparse
import json
import sys

from cpad.core.table import Table
from cpad.models import DiscreteCPAD, EnsembleCPAD, RoutedCPAD
from cpad.modeling.report import build_model


def _make_learner(kind: str):
    if kind == "discrete":
        return DiscreteCPAD(max_lhs=2)
    if kind == "gated":
        from cpad.models.gated import GatedCPAD
        return GatedCPAD()
    if kind == "ensemble":
        from cpad.models.gated import GatedCPAD
        return EnsembleCPAD([DiscreteCPAD(max_lhs=2), GatedCPAD()])
    if kind == "routed":
        return RoutedCPAD()                               # discrete FD + order DC + null-space + marginal
    raise SystemExit(f"unknown learner {kind!r}")


def cmd_analyze(args) -> int:
    table = Table.from_csv(args.path, id_cardinality=args.cap)
    model = _make_learner(args.learner)
    model.fit(table)
    report = build_model(table, model, top_anomalies=args.top)
    print(report.to_text())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n[written] {args.json}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cpad", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="discover constraints and propose a data model")
    a.add_argument("path", help="path to a CSV file")
    a.add_argument("--learner", default="routed",
                   choices=["routed", "discrete", "gated", "ensemble"],
                   help="constraint learner (default: routed = complete CPAD)")
    a.add_argument("--cap", type=int, default=300,
                   help="max column cardinality to treat as modeling column (default 300)")
    a.add_argument("--top", type=int, default=10, help="number of violating cells to show")
    a.add_argument("--json", metavar="FILE", help="also write the model as JSON")
    a.set_defaults(func=cmd_analyze)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
