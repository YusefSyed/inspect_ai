from typing import Literal

from ._common import match_str, str_match_scorer
from ._metrics import accuracy, stderr
from ._scorer import Scorer, scorer


@scorer(metrics=[accuracy(), stderr()])
def match(
    location: Literal["begin", "end", "any", "exact"] = "end",
    *,
    ignore_case: bool = True,
    numeric: bool = False,
) -> Scorer:
    """Scorer which matches text or a number.

    Args:
       location: Location to match at. "any" matches anywhere in the
          output; "exact" requires the output be exactly
          equal to the target (module whitespace, etc.)
       ignore_case: Do case insensitive comparison.
       numeric: Is this a numeric match? When True, currency symbols
          (`$`, `€`, `£`) and formatting markers (`*`, `_`) are stripped
          before numbers are normalized and compared. A comma between
          digits is stripped only inside well-formed groups of three
          digits (`1,234,567`, or lakh style `12,34,567`); otherwise
          (`1,2`, `(3,4)`, `2,5`) the token is not a number, and a target
          such as `1,2,3` is compared as a whole comma-separated list.
          The percent sign is not stripped: `60%` is ambiguous (it could
          mean `60` or `0.6`), so an answer of `60%` will not match a
          numeric target of `60`. To accept a
          percentage-formatted answer, pass both forms as targets, e.g.
          `Target(["60", "60%"])`, where the non-numeric `"60%"` is
          matched as a string.
    """

    def check(value: str, target: str) -> tuple[str, bool]:
        return match_str(
            value=value,
            target=target,
            location=location,
            ignore_case=ignore_case,
            numeric=numeric,
        )

    return str_match_scorer(check)


@scorer(metrics=[accuracy(), stderr()])
def includes(ignore_case: bool = True) -> Scorer:
    """Check whether the specified text is included in the model output.

    Args:
       ignore_case: Use a case insensitive comparison.

    """

    def check(value: str, target: str) -> tuple[str, bool]:
        if ignore_case:
            value = value.casefold()
            target = target.casefold()
        return value, target in value

    return str_match_scorer(check)
