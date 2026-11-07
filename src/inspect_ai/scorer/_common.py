import math
import re
from typing import Callable, Literal

from inspect_ai._util.text import (
    strip_numeric_punctuation,
    strip_punctuation,
)
from inspect_ai.scorer._unicode import unicode_number_to_float
from inspect_ai.solver._task_state import TaskState

from ._metric import CORRECT, INCORRECT, Score
from ._scorer import Scorer
from ._target import Target


def no_response(completion: str) -> bool:
    """Whether the model produced nothing that could be scored.

    An empty or whitespace-only completion never produced output that could
    violate a format, so `no_response` is the correct `ScoreReason` for it
    rather than `invalid_response_format`.

    Args:
        completion: The raw model completion, never an extracted answer.

    Returns:
        True when the raw completion is empty or whitespace only.
    """
    return not completion.strip()


def str_match_scorer(match: Callable[[str, str], tuple[str, bool]]) -> Scorer:
    """Scorer that uses a matching function.

    The matching function returns tuple[str,bool], where str is the answer
    extracted from the model output and bool is whether it matched the target
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Decided once, above the loop, because a raw-empty completion can reach
        # the CORRECT return as well as the INCORRECT one. match() with a target
        # of "." normalizes that target to empty, and "".endswith("") is True;
        # includes() with an empty target is contained in anything. Both score
        # CORRECT off a completion the model never produced. The accepted scope
        # tags a raw-empty completion regardless of the value it ended up with,
        # so the value is preserved here and only the reason is added.
        empty = no_response(state.output.completion or "")
        answer: str | None = None
        for value in target:
            answer, matched = match(state.output.completion, value)
            if matched:
                return Score(
                    value=CORRECT,
                    answer=answer,
                    reason="no_response" if empty else None,
                    explanation=state.output.completion,
                )

        return Score(
            value=INCORRECT,
            answer=answer,
            reason="no_response" if empty else None,
            explanation=state.output.completion,
        )

    return score


def match_str(
    value: str,
    target: str,
    location: Literal["begin", "end", "any", "exact"] = "end",
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
    numeric: bool = False,
) -> tuple[str, bool]:
    # strip ws
    v = value.strip()
    t = target.strip()

    # baseline answer (will only change for numeric)
    answer = v

    # further cleanup
    if ignore_case:
        v = v.casefold()
        t = t.casefold()
    if numeric:
        t = strip_numeric_punctuation(t)
    if numeric and _is_number(t):
        # the target is a number: extract the relevant number(s) from the
        # value, normalize both, and compare for numeric equality. we do
        # NOT fall through to the text comparison below because e.g.
        # "25".endswith("5") is True but 25 != 5.
        v = strip_numeric_punctuation(v)
        t = normalize_number(t)
        words = re.split(r"\s+", v)
        if location == "begin":
            v = first_number_normalized(words)
        elif location == "end":
            words.reverse()
            v = first_number_normalized(words)
        elif location == "exact":
            # exact stays actually exact: no punctuation trimming, so
            # "(42)" does not match a target of "42" under exact
            v = normalize_number(v, trim_punctuation=False)
        else:
            # location == "any": match if any number in the value equals t
            for number in all_numbers_normalized(words):
                if number == t:
                    return number, True
            return answer, False
        answer = v
        return answer, v == t
    elif numeric and (t_list := _normalized_number_list(t)) is not None:
        # the target is a comma-separated list of numbers ("1,2,3"): compare
        # whole list tokens, not raw characters, so "11,2,3" does not end
        # with a match for "1,2,3"
        if location == "exact":
            # the whole value is the list, so "1, 2, 3" is unambiguous
            v = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", v)
            v_list = _normalized_number_list(
                strip_numeric_punctuation(v), trim_punctuation=False
            )
            return answer if v_list is None else v_list, v_list == t_list
        words = re.split(r"\s+", strip_numeric_punctuation(v))
        lists = [x for x in map(_normalized_number_list, words) if x is not None]
        if location == "any":
            return (t_list, True) if t_list in lists else (answer, False)
        if location == "end":
            lists.reverse()
        v_list = lists[0] if lists else ""
        return v_list, v_list == t_list
    elif ignore_punctuation:
        v = strip_punctuation(v)
        t = strip_punctuation(t)

    # comparisons
    if numeric and location != "exact" and _DIGIT_COMMA_DIGIT.search(t):
        # a target with a digit,digit comma ("1,2-dichloroethane") must not
        # match inside a longer number ("11,2-dichloroethane")
        before = r"(?<!\d)(?<!\d[,.])" if t[0].isdigit() else ""
        after = r"(?!\d|[,.]\d)" if t[-1].isdigit() else ""
        if location == "begin":
            before = r"\A"
        elif location == "end":
            after = r"\Z"
        return answer, re.search(before + re.escape(t) + after, v) is not None
    if location == "begin":
        return answer, v.startswith(t)
    elif location == "end":
        return answer, v.endswith(t)
    elif location == "exact":
        return answer, v == t
    else:
        return answer, t in v


_DIGIT_COMMA_DIGIT = re.compile(r"\d,\d")

# Sentence/enclosing punctuation only. Operator characters (~ < > = etc.)
# are deliberately excluded: "<42", ">=42", "~42" express bounds or
# approximations, not the value itself, and trimming them would grade a
# hedge as the exact answer (false CORRECTs).
_NUMERIC_PUNCTUATION_TRIM = "!?:;()[]{}'\"`"


def _clean_numeric_word(s: str) -> str:
    return s.strip(_NUMERIC_PUNCTUATION_TRIM)


def _parse_number(s: str, *, trim_punctuation: bool = True) -> float | None:
    """Parse `s` as a finite number, or None.

    Recognises plain float syntax (signs, decimals, exponents) and the
    full set of inputs handled by ``unicode_number_to_float`` (unicode
    minus, vulgar fractions, superscripts, Chinese numerals, fullwidth
    digits, locale grouping). Rejects strings with trailing non-numeric
    content (so ``"5 some text"`` returns None) and any ASCII comma, which
    ``strip_numeric_punctuation`` leaves only between digits outside a
    thousands grouping (so ``"1,2"`` is not read as a locale decimal).
    Sentence/enclosing punctuation attached to the number ("42!", "(42)")
    is trimmed unless ``trim_punctuation=False`` — the ``location="exact"``
    path passes False so that exact stays actually exact. ``nan`` and
    ``inf`` also return None.
    """
    s_cleaned = _clean_numeric_word(s) if trim_punctuation else s
    # a comma left after strip_numeric_punctuation sits between two digits
    # outside a thousands grouping ("1,2"): not one number
    if "," in s_cleaned:
        return None
    for parse in (float, unicode_number_to_float):
        try:
            num = parse(s_cleaned)
        except ValueError:
            continue
        if math.isfinite(num):
            return num
    return None


def _is_number(s: str) -> bool:
    return _parse_number(s) is not None


def _normalized_number_list(s: str, *, trim_punctuation: bool = True) -> str | None:
    """Normalize a comma-separated list of two or more numbers, or None."""
    if trim_punctuation:
        s = _clean_numeric_word(s)
    parts = s.split(",")
    if len(parts) < 2:
        return None
    normalized: list[str] = []
    for part in parts:
        num = _parse_number(part, trim_punctuation=False)
        if num is None:
            return None
        normalized.append(format(num, ".5g"))
    return ",".join(normalized)


def first_number_normalized(words: list[str]) -> str:
    number = next((word for word in words if _is_number(word)), None)
    return normalize_number(number) if number is not None else ""


def all_numbers_normalized(words: list[str]) -> list[str]:
    return [normalize_number(word) for word in words if _is_number(word)]


def normalize_number(
    number: str, precision: int = 5, *, trim_punctuation: bool = True
) -> str:
    num = _parse_number(number, trim_punctuation=trim_punctuation)
    if num is None:
        return number
    return format(num, f".{precision}g")
