from pathlib import Path

import pandas as pd
import pyarrow as pa
from pydantic import JsonValue

from inspect_ai import Task, eval
from inspect_ai.analysis import (
    EvalModel,
    EventInfo,
    EventTiming,
    SampleColumn,
    ToolEventColumns,
    events_df,
    messages_df,
    samples_df,
)
from inspect_ai.analysis._dataframe.samples.columns import (
    SampleMessages,
    SampleSummary,
)
from inspect_ai.analysis._dataframe.samples.table import SAMPLE_ID
from inspect_ai.dataset import Sample
from inspect_ai.event import Event
from inspect_ai.model import ChatMessage

LOGS_DIR = Path(__file__).parent / "test_logs"
SECURITY_GUIDE_LOG = LOGS_DIR / "2025-05-12T20-28-26-04-00_security-guide.json"
BROWSER_LOG = LOGS_DIR / "2025-05-12T20-27-36-04-00_browser.json"

EVENT_ID = "event_id"
MESSAGE_ID = "message_id"


def test_samples_df_parallel() -> None:
    dfs = samples_df(LOGS_DIR)
    dfs_parallel = samples_df(LOGS_DIR, parallel=2)
    assert dfs[SAMPLE_ID].to_list() == dfs_parallel[SAMPLE_ID].to_list()


def test_samples_df_messages_parallel() -> None:
    columns = SampleSummary + SampleMessages
    dfs = samples_df(LOGS_DIR, columns=columns)
    dfs_parallel = samples_df(LOGS_DIR, columns=columns, parallel=2)
    assert dfs[SAMPLE_ID].to_list() == dfs_parallel[SAMPLE_ID].to_list()


def test_events_df_parallel() -> None:
    dfs = events_df(LOGS_DIR)
    dfs_parallel = events_df(LOGS_DIR, parallel=2)
    assert dfs[EVENT_ID].to_list() == dfs_parallel[EVENT_ID].to_list()


def test_messages_df_parallel() -> None:
    dfs = messages_df(LOGS_DIR)
    dfs_parallel = messages_df(LOGS_DIR, parallel=2)
    assert dfs[MESSAGE_ID].to_list() == dfs_parallel[MESSAGE_ID].to_list()


# filters for the parallel readers must be picklable (module level)
def _is_tool_event(event: Event) -> bool:
    return event.event == "tool"


def _is_tool_message(message: ChatMessage) -> bool:
    return message.role == "tool"


def test_events_df_parallel_filter_matches_one_log_of_two() -> None:
    # the security guide log has no tool events, so its worker yields no rows
    logs = [str(SECURITY_GUIDE_LOG), str(BROWSER_LOG)]
    columns = EvalModel + SampleSummary + EventInfo + EventTiming + ToolEventColumns
    dfs = events_df(logs, columns=columns, filter=_is_tool_event)
    dfs_parallel = events_df(logs, columns=columns, filter=_is_tool_event, parallel=2)
    assert len(dfs_parallel) == 4
    assert dfs[EVENT_ID].to_list() == dfs_parallel[EVENT_ID].to_list()
    # the parallel concat may order a log's wildcard (score) columns differently
    assert sorted(dfs_parallel.columns) == sorted(dfs.columns)
    mismatched = {
        col: (str(dfs[col].dtype), str(dfs_parallel[col].dtype))
        for col in dfs.columns
        if dfs[col].dtype != dfs_parallel[col].dtype
    }
    assert mismatched == {}, f"(serial, parallel) dtypes differ: {mismatched}"


def test_messages_df_parallel_filter_matches_nothing() -> None:
    dfs = messages_df(SECURITY_GUIDE_LOG, filter=_is_tool_message)
    dfs_parallel = messages_df(SECURITY_GUIDE_LOG, filter=_is_tool_message, parallel=2)
    assert len(dfs_parallel) == 0
    assert list(dfs_parallel.columns) == list(dfs.columns)
    assert MESSAGE_ID in dfs_parallel.columns


def test_samples_df_parallel_reconciles_string_columns(tmp_path: Path) -> None:
    """A column read as strings from one log is strings for every log."""
    logs: list[list[dict[str, JsonValue]]] = [
        [{"big": 1, "mixed": "C"}],
        [{}, {"big": 2**63 + 1, "mixed": 0.5, "note": "x"}],
    ]
    for metadata in logs:
        eval(
            Task(dataset=[Sample(input="hi", metadata=m) for m in metadata]),
            model="mockllm/model",
            log_dir=str(tmp_path),
            display="none",
        )
    columns = SampleSummary + [
        SampleColumn("big", path="metadata.big"),
        SampleColumn("note", path="metadata.note"),
    ]

    def read(parallel: bool | int) -> pd.DataFrame:
        df = samples_df(tmp_path, columns=columns, parallel=parallel)
        return df.sort_values(SAMPLE_ID, ignore_index=True)

    serial, parallel = read(False), read(2)
    for column in ["big", "note", "metadata_big", "metadata_mixed"]:
        pd.testing.assert_series_equal(parallel[column], serial[column])
    assert sorted(parallel["big"].dropna()) == ["1", str(2**63 + 1)]
    assert sorted(serial["metadata_mixed"].dropna()) == ["0.5", "C"]
    pa.Table.from_pandas(parallel)
