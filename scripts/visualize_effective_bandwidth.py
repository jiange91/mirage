#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


EVENT_RE = re.compile(r"^(?P<task_name>.+)_(?P<task_index>\d+)_(?P<counter>\d+)$")

DTYPE_SIZES = {
    920: 1,  # float4 packed approximation
    925: 1,  # int4 packed approximation
    926: 1,  # uint4 packed approximation
    930: 1,  # float8
    935: 1,  # int8
    936: 1,  # uint8
    940: 2,  # float16
    941: 2,  # bfloat16
    945: 2,  # int16
    946: 2,  # uint16
    950: 4,  # float32
    955: 4,  # int32
    956: 4,  # uint32
    960: 8,  # float64
    965: 8,  # int64
    966: 8,  # uint64
}

TASK_TYPE_TO_NAME = {
    10: "TASK_BEGIN_TASK_GRAPH",
    101: "TASK_EMBEDDING",
    102: "TASK_RMS_NORM_LINEAR",
    103: "TASK_ATTENTION_1",
    104: "TASK_ATTENTION_2",
    105: "TASK_SILU_MUL_LINEAR",
    106: "TASK_ALLREDUCE",
    107: "TASK_REDUCE",
    108: "TASK_LINEAR_WITH_RESIDUAL",
    109: "TASK_ARGMAX",
    110: "TASK_ARGMAX_PARTIAL",
    111: "TASK_ARGMAX_REDUCE",
    112: "TASK_FIND_NGRAM_PARTIAL",
    113: "TASK_FIND_NGRAM_GLOBAL",
    114: "TASK_TARGET_VERIFY_GREEDY",
    115: "TASK_SINGLE_BATCH_EXTEND_ATTENTION",
    116: "TASK_PAGED_ATTENTION_1",
    117: "TASK_PAGED_ATTENTION_2",
    118: "TASK_SILU_MUL",
    119: "TASK_RMS_NORM",
    120: "TASK_LINEAR",
    121: "TASK_IDENTITY",
    150: "TASK_HOPPER_TASK_BEGIN",
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",
    152: "TASK_LINEAR_HOPPER",
    153: "TASK_PAGED_ATTENTION_HOPPER",
    154: "TASK_RMS_NORM_HOPPER",
    155: "TASK_LINEAR_SWAPAB_HOPPER",
    156: "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER",
    157: "TASK_LINEAR_CUTLASS_HOPPER",
    158: "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER",
    159: "TASK_SILU_MUL_HOPPER",
    160: "TASK_EMBEDDING_HOPPER",
    161: "TASK_MOE_W13_LINEAR_SM90",
    162: "TASK_MOE_W2_LINEAR_SM90",
    163: "TASK_SPLITK_LINEAR_SWAPAB_HOPPER",
    198: "TASK_HOPPER_TASK_END",
    200: "TASK_SCHD_TASKS",
    201: "TASK_SCHD_EVENTS",
    202: "TASK_GET_EVENT",
    203: "TASK_GET_NEXT_TASK",
    230: "TASK_SM100_TASK_BEGIN",
    251: "TASK_SPLITK_LINEAR_SM100",
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",
    253: "TASK_LINEAR_SM100",
    254: "TASK_MOE_W13_LINEAR_SM100",
    255: "TASK_MOE_W2_LINEAR_SM100",
    257: "TASK_ATTN_SM100",
    258: "TASK_ARGMAX_REDUCE_SM100",
    259: "TASK_ARGMAX_PARTIAL_SM100",
    260: "TASK_MOE_TOPK_SOFTMAX_SM100",
    261: "TASK_MOE_MUL_SUM_ADD_SM100",
    262: "TASK_TENSOR_INIT",
    298: "TASK_SM100_TASK_END",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
}

SKIP_TASK_NAMES = {
    "TASK_BEGIN_TASK_GRAPH",
    "TASK_SCHD_TASKS",
    "TASK_SCHD_EVENTS",
    "TASK_GET_EVENT",
    "TASK_GET_NEXT_TASK",
}


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def tensor_logical_bytes(tensor_desc):
    dtype = int(tensor_desc["data_type"])
    if dtype == 999:
        return 0
    if dtype not in DTYPE_SIZES:
        return 0
    return _product(tensor_desc["dims"]) * DTYPE_SIZES[dtype]


def task_logical_bytes(task):
    inputs = task.get("inputs") or []
    outputs = task.get("outputs") or []
    input_bytes = sum(tensor_logical_bytes(t) for t in inputs)
    output_bytes = sum(tensor_logical_bytes(t) for t in outputs)
    return input_bytes, output_bytes, input_bytes + output_bytes


def find_task_list(obj):
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            if all("task_type" in x and "inputs" in x and "outputs" in x for x in obj):
                return obj
        for item in obj:
            result = find_task_list(item)
            if result is not None:
                return result
    elif isinstance(obj, dict):
        for key in ("tasks", "all_tasks"):
            value = obj.get(key)
            result = find_task_list(value)
            if result is not None:
                return result
        for value in obj.values():
            result = find_task_list(value)
            if result is not None:
                return result
    return None


def load_task_graph(path):
    with open(path) as f:
        data = json.load(f)
    tasks = find_task_list(data)
    if tasks is None:
        raise ValueError(f"Could not find task list in {path}")
    return tasks


def parse_trace_events_from_json(path):
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "traceEvents" in data:
        trace_events = data["traceEvents"]
    elif isinstance(data, list):
        trace_events = data
    else:
        raise ValueError(f"Unsupported JSON trace structure in {path}")

    events = []
    for event in trace_events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        match = EVENT_RE.match(name)
        if match is None:
            continue

        phase = event.get("ph")
        if phase == "X":
            ts = float(event["ts"])
            dur = float(event["dur"])
        else:
            continue

        events.append(
            {
                "name": name,
                "task_name": match.group("task_name"),
                "task_index": int(match.group("task_index")),
                "counter": int(match.group("counter")),
                "ts_us": ts,
                "dur_us": dur,
                "track_id": event.get("tid"),
                "block_name": None,
                "group_name": None,
            }
        )
    return events


def parse_trace_events_from_perfetto(path):
    try:
        from perfetto.trace_processor import TraceProcessor
    except Exception as exc:
        raise RuntimeError(
            "Binary .perfetto-trace parsing requires the Perfetto Python package. "
            "Install `perfetto` or export the trace as JSON and rerun."
        ) from exc

    tp = TraceProcessor(file_path=str(path))
    rows = tp.query(
        """
        SELECT
          s.name AS name,
          s.ts / 1000.0 AS ts_us,
          s.dur / 1000.0 AS dur_us,
          s.track_id AS track_id,
          COALESCE(t.name, '') AS track_name,
          t.parent_id AS parent_id,
          COALESCE(parent.name, '') AS parent_track_name,
          COALESCE(block_track.name, '') AS block_track_name
        FROM slice s
        LEFT JOIN track t ON s.track_id = t.id
        LEFT JOIN track parent ON t.parent_id = parent.id
        LEFT JOIN track block_track ON parent.id = block_track.id + 1
        WHERE s.dur >= 0
        """
    )

    events = []
    for row in rows:
        name = row.name
        if not isinstance(name, str):
            continue
        match = EVENT_RE.match(name)
        if match is None:
            continue
        names = [row.track_name or None, row.parent_track_name or None]
        group_name = next(
            (name for name in names if isinstance(name, str) and name.startswith("group_")),
            None,
        )
        block_name = row.block_track_name or None
        events.append(
            {
                "name": name,
                "task_name": match.group("task_name"),
                "task_index": int(match.group("task_index")),
                "counter": int(match.group("counter")),
                "ts_us": float(row.ts_us),
                "dur_us": float(row.dur_us),
                "track_id": row.track_id,
                "block_name": block_name,
                "group_name": group_name,
            }
        )
    return events


def load_trace_events(path):
    path = Path(path)
    with open(path, "rb") as f:
        prefix = f.read(1)
    if prefix in (b"{", b"["):
        return parse_trace_events_from_json(path)
    return parse_trace_events_from_perfetto(path)


def filter_events(events, block=None, group=None, t_min_us=None, t_max_us=None):
    filtered = []
    for event in events:
        if block is not None and event.get("block_name") != f"block_{block}":
            continue
        if group is not None and event.get("group_name") != f"group_{group}":
            continue
        start_us = event["ts_us"]
        end_us = event["ts_us"] + event["dur_us"]
        if t_min_us is not None and end_us <= t_min_us:
            continue
        if t_max_us is not None and start_us >= t_max_us:
            continue

        clipped = dict(event)
        if t_min_us is not None and clipped["ts_us"] < t_min_us:
            clipped_end = clipped["ts_us"] + clipped["dur_us"]
            clipped["ts_us"] = t_min_us
            clipped["dur_us"] = max(0.0, clipped_end - t_min_us)
        if t_max_us is not None:
            clipped["dur_us"] = max(
                0.0, min(clipped["ts_us"] + clipped["dur_us"], t_max_us) - clipped["ts_us"]
            )
        if clipped["dur_us"] <= 0:
            continue
        filtered.append(clipped)
    return filtered


def summarize_track_labels(events, limit=20):
    block_counts = Counter(event.get("block_name") for event in events)
    group_counts = Counter(event.get("group_name") for event in events)
    block_summary = [
        {"label": label, "count": count}
        for label, count in block_counts.most_common(limit)
    ]
    group_summary = [
        {"label": label, "count": count}
        for label, count in group_counts.most_common(limit)
    ]
    return block_summary, group_summary


def join_events_with_tasks(events, tasks):
    joined = []
    warnings = Counter()
    warning_examples = defaultdict(list)
    for event in events:
        if event["task_name"] in SKIP_TASK_NAMES:
            warnings["skipped_meta_task"] += 1
            if len(warning_examples["skipped_meta_task"]) < 5:
                warning_examples["skipped_meta_task"].append(event["name"])
            continue
        task_index = event["task_index"]
        if task_index < 0 or task_index >= len(tasks):
            warnings["task_index_out_of_range"] += 1
            if len(warning_examples["task_index_out_of_range"]) < 5:
                warning_examples["task_index_out_of_range"].append(event["name"])
            continue
        task = tasks[task_index]
        input_bytes, output_bytes, total_bytes = task_logical_bytes(task)
        if total_bytes == 0:
            warnings["zero_logical_bytes"] += 1
            if len(warning_examples["zero_logical_bytes"]) < 5:
                warning_examples["zero_logical_bytes"].append(
                    f'{event["name"]} -> task_type={task.get("task_type")} variant={task.get("variant_id")}'
                )

        json_task_type = task.get("task_type")
        task_name_from_json = event["task_name"]
        if isinstance(json_task_type, int):
            # Optional consistency check based on the stable runtime enum names encoded in the trace.
            # We only warn; some scheduler/meta tasks may still be useful with zero bytes.
            expected_name = TASK_TYPE_TO_NAME.get(json_task_type)
            if expected_name is not None and expected_name != task_name_from_json:
                warnings["task_type_name_mismatch"] += 1
                if len(warning_examples["task_type_name_mismatch"]) < 5:
                    warning_examples["task_type_name_mismatch"].append(
                        f'{event["name"]} -> json_task_type={json_task_type} ({expected_name})'
                    )

        duration_us = event["dur_us"]
        duration_s = duration_us * 1e-6
        bandwidth_gbps = (total_bytes / duration_s) / 1e9 if duration_s > 0 else math.inf
        joined.append(
            {
                **event,
                "json_task_type": json_task_type,
                "variant_id": task.get("variant_id"),
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "total_bytes": total_bytes,
                "bandwidth_gbps": bandwidth_gbps,
            }
        )
    return joined, warnings, warning_examples


def write_event_csv(events, path):
    fieldnames = [
        "name",
        "task_name",
        "task_index",
        "counter",
        "track_id",
        "ts_us",
        "dur_us",
        "json_task_type",
        "variant_id",
        "input_bytes",
        "output_bytes",
        "total_bytes",
        "bandwidth_gbps",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def build_bandwidth_series(events, bin_us):
    min_time_us = min(event["ts_us"] for event in events)
    normalized_events = [
        {
            **event,
            "ts_us": event["ts_us"] - min_time_us,
        }
        for event in events
    ]
    end_time_us = max(event["ts_us"] + event["dur_us"] for event in normalized_events)
    num_bins = max(1, math.ceil(end_time_us / bin_us))
    total_bytes_per_bin = [0.0] * num_bins
    read_bytes_per_bin = [0.0] * num_bins
    write_bytes_per_bin = [0.0] * num_bins

    for event in normalized_events:
        start_us = event["ts_us"]
        end_us = event["ts_us"] + event["dur_us"]
        if end_us <= start_us:
            continue
        start_bin = max(0, int(start_us // bin_us))
        end_bin = min(num_bins - 1, int((end_us - 1e-12) // bin_us))
        duration_us = end_us - start_us
        for bin_idx in range(start_bin, end_bin + 1):
            bin_start = bin_idx * bin_us
            bin_end = bin_start + bin_us
            overlap_us = max(0.0, min(end_us, bin_end) - max(start_us, bin_start))
            if overlap_us <= 0:
                continue
            ratio = overlap_us / duration_us
            total_bytes_per_bin[bin_idx] += event["total_bytes"] * ratio
            read_bytes_per_bin[bin_idx] += event["input_bytes"] * ratio
            write_bytes_per_bin[bin_idx] += event["output_bytes"] * ratio

    x_us = [(idx + 0.5) * bin_us for idx in range(num_bins)]
    scale = 1e6 / bin_us  # bytes / us -> bytes / s
    total_gbps = [(value * scale) / 1e9 for value in total_bytes_per_bin]
    read_gbps = [(value * scale) / 1e9 for value in read_bytes_per_bin]
    write_gbps = [(value * scale) / 1e9 for value in write_bytes_per_bin]
    return x_us, total_gbps, read_gbps, write_gbps


def write_summary(events, path):
    by_task = defaultdict(lambda: {"count": 0, "total_bytes": 0, "total_us": 0.0})
    for event in events:
        summary = by_task[event["task_name"]]
        summary["count"] += 1
        summary["total_bytes"] += event["total_bytes"]
        summary["total_us"] += event["dur_us"]

    payload = []
    for task_name, summary in sorted(by_task.items()):
        duration_s = summary["total_us"] * 1e-6
        avg_gbps = (summary["total_bytes"] / duration_s) / 1e9 if duration_s > 0 else math.inf
        payload.append(
            {
                "task_name": task_name,
                "count": summary["count"],
                "total_bytes": summary["total_bytes"],
                "total_us": summary["total_us"],
                "average_effective_bandwidth_gbps": avg_gbps,
            }
        )

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_bandwidth(x_us, total_gbps, read_gbps, write_gbps, out_path, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_us, total_gbps, label="total", linewidth=1.5)
    ax.plot(x_us, read_gbps, label="read", linewidth=1.2)
    ax.plot(x_us, write_gbps, label="write", linewidth=1.2)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Join Mirage task_graph JSON with trace events and plot effective bandwidth over time."
    )
    parser.add_argument("--trace", required=True, help="Path to .perfetto-trace or JSON trace")
    parser.add_argument("--task-graph", required=True, help="Path to task_graph_*.json")
    parser.add_argument("--bin-us", type=float, default=0.5, help="Time bin width in microseconds")
    parser.add_argument("--block", type=int, default=None, help="Filter to block_N in the trace")
    parser.add_argument("--group", type=int, default=None, help="Filter to group_N in the trace")
    parser.add_argument("--t-min-us", type=float, default=None, help="Keep events overlapping [t-min-us, t-max-us)")
    parser.add_argument("--t-max-us", type=float, default=None, help="Keep events overlapping [t-min-us, t-max-us)")
    parser.add_argument("--list-tracks", action="store_true", help="Print available block/group labels and exit")
    parser.add_argument(
        "--out-prefix",
        default="effective_bandwidth",
        help="Output prefix for CSV/JSON/PNG artifacts",
    )
    args = parser.parse_args()

    tasks = load_task_graph(args.task_graph)
    all_events = load_trace_events(args.trace)
    if args.list_tracks:
        block_summary, group_summary = summarize_track_labels(all_events)
        print("Block labels:")
        for item in block_summary:
            print(f"  {item['label']}: {item['count']}")
        print("Group labels:")
        for item in group_summary:
            print(f"  {item['label']}: {item['count']}")
        return

    events = filter_events(
        all_events,
        block=args.block,
        group=args.group,
        t_min_us=args.t_min_us,
        t_max_us=args.t_max_us,
    )
    if not events:
        block_summary, group_summary = summarize_track_labels(all_events)
        raise RuntimeError(
            "No events matched the requested filters. "
            f"Available block labels (top {len(block_summary)}): {block_summary}. "
            f"Available group labels (top {len(group_summary)}): {group_summary}."
        )
    joined_events, warnings, warning_examples = join_events_with_tasks(events, tasks)
    if not joined_events:
        raise RuntimeError(
            "No joinable events found. Make sure the trace contains names like "
            "{task_name}_{task_index}_{counter}."
        )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".events.csv")
    summary_path = out_prefix.with_suffix(".summary.json")
    png_path = out_prefix.with_suffix(".png")

    write_event_csv(joined_events, csv_path)
    write_summary(joined_events, summary_path)
    x_us, total_gbps, read_gbps, write_gbps = build_bandwidth_series(joined_events, args.bin_us)
    title = f"Effective bandwidth over time (bin={args.bin_us} us)"
    if args.block is not None:
        title += f", block={args.block}"
    if args.group is not None:
        title += f", group={args.group}"
    if args.t_min_us is not None or args.t_max_us is not None:
        title += f", window=[{args.t_min_us or 0}, {args.t_max_us or 'end'}) us"

    plot_bandwidth(
        x_us,
        total_gbps,
        read_gbps,
        write_gbps,
        png_path,
        title=title,
    )

    task_counts = Counter(event["task_name"] for event in joined_events)
    print(f"Loaded {len(tasks)} tasks from {args.task_graph}")
    print(f"Joined {len(joined_events)} trace events from {args.trace}")
    print(f"Task types in joined events: {len(task_counts)}")
    print(f"Wrote event CSV: {csv_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print(f"Wrote bandwidth plot: {png_path}")
    if warnings:
        print("Warnings:", file=sys.stderr)
        for key, count in sorted(warnings.items()):
            print(f"  {key}: {count}", file=sys.stderr)
            for example in warning_examples.get(key, []):
                print(f"    example: {example}", file=sys.stderr)


if __name__ == "__main__":
    main()
