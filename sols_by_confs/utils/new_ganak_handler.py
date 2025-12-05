"""Ganak runner + parser for Linux.

Features
- Run the local `./ganak` (or a provided path) on a DIMACS CNF file
- Parse Ganak stdout into a strongly-typed Python result with scientifically relevant fields

The parser is designed to be robust to minor log-format variations observed in Ganak/Arjun.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# for a fix with the clause distribution
from dataclasses import dataclass, field

__all__ = [
    "run_ganak",
    "run_ganak_text",
    "run_ganak_parsed",
    "parse_ganak_output",
    "GanakResult",
    "ClauseDistribution",
]


# -------------------------------
# Exceptions
# -------------------------------
class GanakNotFoundError(FileNotFoundError):
    pass


class GanakPermissionError(PermissionError):
    pass


# -------------------------------
# Runner
# -------------------------------

def _ensure_executable(path: Path) -> None:
    if not path.exists():
        raise GanakNotFoundError(f"ganak binary not found at: {path}")
    if not os.access(path, os.X_OK):
        raise GanakPermissionError(
            f"ganak binary is not executable: {path}. Run `chmod +x {path}."
        )


def run_ganak(
    cnf_path: Path | str,
    *,
    ganak_path: Path | str = Path("./ganak"),
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    cwd: Optional[Path | str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Invoke `ganak` on a CNF file and return the CompletedProcess.

    Parameters
    ----------
    cnf_path : Path | str
        Path to the input CNF file.
    ganak_path : Path | str, default './ganak'
        Path to the ganak binary.
    extra_args : iterable of str, optional
        Additional CLI args to pass to ganak.
    timeout : float, optional
        Seconds to wait before raising `TimeoutExpired`.
    cwd : Path | str, optional
        Working directory for the process (defaults to current).
    env : dict, optional
        Environment variables for the process.
    check : bool, default True
        If True, raise `CalledProcessError` on non-zero exit.

    Returns
    -------
    subprocess.CompletedProcess
        Process with `args`, `returncode`, `stdout`, `stderr`.
    """
    cnf_path = Path(cnf_path)
    ganak_path = Path(ganak_path)

    if not cnf_path.exists():
        raise FileNotFoundError(f"CNF file not found: {cnf_path}")

    _ensure_executable(ganak_path)

    cmd = [str(ganak_path), str(cnf_path)]
    if extra_args:
        cmd.extend(list(extra_args))

    proc = subprocess.run(
        cmd,
        cwd=None if cwd is None else str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )
    return proc


def run_ganak_text(
    cnf_path: Path | str,
    *,
    ganak_path: Path | str = Path("./ganak"),
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    cwd: Optional[Path | str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
) -> str:
    """Convenience wrapper returning only stdout as a string."""
    return run_ganak(
        cnf_path,
        ganak_path=ganak_path,
        extra_args=extra_args,
        timeout=timeout,
        cwd=cwd,
        env=env,
        check=check,
    ).stdout


# -------------------------------
# Parsed result types
# -------------------------------
@dataclass
class ClauseDistribution:
    bin_irred: Optional[int] = None
    bin_red: Optional[int] = None
    long_irred_non_ternary: Optional[int] = None
    long_irred_ternary: Optional[int] = None
    long_red_non_ternary: Optional[int] = None
    long_red_ternary: Optional[int] = None

    @property
    def nclauses_irredundant(self) -> Optional[int]:
        parts = [self.bin_irred, self.long_irred_non_ternary, self.long_irred_ternary]
        if any(p is None for p in parts):
            return None
        return int(self.bin_irred) + int(self.long_irred_non_ternary) + int(self.long_irred_ternary)

    @property
    def nclauses_total(self) -> Optional[int]:
        parts = [
            self.bin_irred,
            self.bin_red,
            self.long_irred_non_ternary,
            self.long_irred_ternary,
            self.long_red_non_ternary,
            self.long_red_ternary,
        ]
        if any(p is None for p in parts):
            return None
        return int(self.bin_irred) + int(self.bin_red) + int(self.long_irred_non_ternary) + int(self.long_irred_ternary) + int(self.long_red_non_ternary) + int(self.long_red_ternary)


@dataclass
class GanakResult:
    # Identity
    cnf_file: str

    # SAT / count
    satisfiable: Optional[bool] = None
    model_count_exact: Optional[int] = None  # Python int is arbitrary precision
    log10_estimate: Optional[float] = None

    # Guarantees (ApproxMC-style)
    pac_epsilon: Optional[float] = None
    pac_delta: Optional[float] = None

    # Timing / resources
    runtime_total: Optional[float] = None  # seconds, Total time [Arjun+GANAK]
    memory_gb: Optional[float] = None

    # Structure / sets
    nvars: Optional[int] = None
    projection_set_size: Optional[int] = None
    sampling_set_size: Optional[int] = None
    opt_sampling_set_size: Optional[int] = None
    ind_size: Optional[int] = None
    independence_distribution: Optional[str] = None  # string over {I,O,N}
    disconnected_components: Optional[int] = None

    # Simplification
    simplification_removed: Optional[int] = None
    simplification_removed_pct: Optional[float] = None

    # Clauses
    # clause_dist: ClauseDistribution = ClauseDistribution()
    clause_dist: ClauseDistribution = field(default_factory=ClauseDistribution)

    # Search effort
    decisions: Optional[int] = None
    conflicts: Optional[int] = None

    # Raw output for traceability
    raw_stdout: str = ""


# -------------------------------
# Parser
# -------------------------------
_re_float = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_re_int = r"([+-]?\d+)"


def _search(pattern: str, text: str) -> Optional[re.Match]:
    return re.search(pattern, text, flags=re.MULTILINE)


def parse_ganak_output(stdout: str, *, cnf_file: str) -> GanakResult:
    """Parse Ganak stdout into a structured `GanakResult`.

    Notes on *clause counts*:
        Ganak reports a distribution that may reflect the *post-simplification* working formula.
        We expose both irredundant and (if present) redundant counts. When the original clause
        count is required, obtain it from the DIMACS header ('p cnf n m') by reading the CNF file.
    """
    res = GanakResult(cnf_file=str(cnf_file), raw_stdout=stdout)

    # SAT/UNSAT
    if re.search(r"^s\s+SATISFIABLE\b", stdout, flags=re.MULTILINE):
        res.satisfiable = True
    elif re.search(r"^s\s+UNSATISFIABLE\b", stdout, flags=re.MULTILINE):
        res.satisfiable = False

    # Exact model count (arbitrary integer)
    m = _search(r"^c\s+s\s+exact\s+arb\s+int\s+" + _re_int + r"\s*$", stdout)
    if m:
        try:
            res.model_count_exact = int(m.group(1))
        except Exception:
            pass

    # log10-estimate
    m = _search(r"^c\s+s\s+log10-estimate\s+" + _re_float + r"\s*$", stdout)
    if m:
        res.log10_estimate = float(m.group(1))

    # PAC guarantees
    m = _search(r"^c\s+s\s+pac\s+guarantees\s+epsilon:\s*" + _re_float + r"\s+delta:\s*" + _re_float + r"\s*$", stdout)
    if m:
        res.pac_epsilon = float(m.group(1))
        res.pac_delta = float(m.group(2))

    # Runtime (Total time [Arjun+GANAK])
    m = _search(r"Total\s+time\s*\[[^\]]*\]:\s*" + _re_float + r"\b", stdout)
    if m:
        res.runtime_total = float(m.group(1))

    # Memory
    m = _search(r"Mem\s+used\s+" + _re_float + r"\s*GB", stdout)
    if m:
        res.memory_gb = float(m.group(1))

    # Vars / projection
    m = _search(r"CNF\s+projection\s+set\s+size:\s*" + _re_int, stdout)
    if m:
        res.projection_set_size = int(m.group(1))

    # Independent support and total vars
    m = _search(r"ind size:\s*" + _re_int + r"\s+nvars:\s*" + _re_int, stdout)
    if m:
        res.ind_size = int(m.group(1))
        res.nvars = int(m.group(2))

    # Sampling set sizes
    m = _search(r"Sampling\s+set\s+size:\s*" + _re_int, stdout)
    if m:
        res.sampling_set_size = int(m.group(1))

    m = _search(r"Opt\s+sampling\s+set\s+size:\s*" + _re_int, stdout)
    if m:
        res.opt_sampling_set_size = int(m.group(1))

    # Independence distribution
    m = _search(r"indep/optional/none\s+distribution:\s*([ION]+)", stdout)
    if m:
        res.independence_distribution = m.group(1)

    # Disconnected components
    m = _search(r"Found\s+" + _re_int + r"\s+disconnected\s+component\(s\)", stdout)
    if m:
        res.disconnected_components = int(m.group(1))

    # Simplification removed
    m = _search(r"\[arjun\][^\n]*removed:\s*" + _re_int + r"\s+perc:\s*" + _re_float, stdout)
    if m:
        res.simplification_removed = int(m.group(1))
        res.simplification_removed_pct = float(m.group(2))

    # Clause distribution
    m = _search(r"Bin\s+irred/red\s+" + _re_int + r"\s+" + _re_int, stdout)
    if m:
        res.clause_dist.bin_irred = int(m.group(1))
        res.clause_dist.bin_red = int(m.group(2))

    m = _search(r"Long\s+irred\s+cls/tri\s+" + _re_int + r"\s+" + _re_int, stdout)
    if m:
        res.clause_dist.long_irred_non_ternary = int(m.group(1))
        res.clause_dist.long_irred_ternary = int(m.group(2))

    m = _search(r"Long\s+red\s+cls/tri\s+" + _re_int + r"\s+" + _re_int, stdout)
    if m:
        res.clause_dist.long_red_non_ternary = int(m.group(1))
        res.clause_dist.long_red_ternary = int(m.group(2))

    # Decisions & conflicts
    m = _search(r"decisions\s+K\s+" + _re_int, stdout)
    if m:
        res.decisions = int(m.group(1))

    m = _search(r"^c\s+o\s+conflicts\s+" + _re_int, stdout)
    if m:
        res.conflicts = int(m.group(1))
    else:
        # Backup pattern without the leading 'c o'
        m2 = _search(r"^conflicts\s+" + _re_int, stdout)
        if m2:
            res.conflicts = int(m2.group(1))

    return res


# -------------------------------
# High-level convenience
# -------------------------------

def run_ganak_parsed(
    cnf_path: Path | str,
    *,
    ganak_path: Path | str = Path("./ganak"),
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    cwd: Optional[Path | str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
) -> GanakResult:
    """Run ganak and return a parsed `GanakResult`.

    Notes
    -----
    - The `cnf_file` field is taken from the provided `cnf_path` (basename).
    - If you need the **original** (DIMACS-declared) number of variables/clauses,
      parse the CNF header ('p cnf n m') separately. The clause distribution here
      reflects Ganak's reported *working formula* after simplifications.
    """
    cnf_path = Path(cnf_path)
    stdout = run_ganak_text(
        cnf_path,
        ganak_path=ganak_path,
        extra_args=extra_args,
        timeout=timeout,
        cwd=cwd,
        env=env,
        check=check,
    )
    return parse_ganak_output(stdout, cnf_file=str(cnf_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Invoke ganak on a CNF file.")
    parser.add_argument("--cnf", type=str, help="Path to input CNF file")
    parser.add_argument(
        "--ganak",
        type=str,
        default="./ganak",
        help="Path to the ganak binary (default: ./ganak)",
    )
    parser.add_argument(
        "--",
        dest="extra",
        nargs=argparse.REMAINDER,
        help="Additional args passed through to ganak",
    )

    args = parser.parse_args()
    result = run_ganak_parsed(args.cnf, ganak_path=args.ganak, extra_args=args.extra)
    # Print a compact, human-readable summary
    print(
        f"file={Path(result.cnf_file).name}\tSAT={result.satisfiable}\t#models={result.model_count_exact}\t"
        f"log10≈{result.log10_estimate}\tT_total={result.runtime_total}s\tvars={result.nvars}\t"
        f"S={result.sampling_set_size}/optS={result.opt_sampling_set_size}\tind={result.ind_size}\t"
        f"clauses_irred={result.clause_dist.nclauses_irredundant}\tmem={result.memory_gb}GB"
    )
