"""Export step-ordered V RGB videos, subtitles, provenance and pending reviews.

No simulator or model is called. A completed read-only batch audit supplies the
navigation evidence; its full source snapshots must still match. MP4 validation
checks encoding metadata and hashes, not human playback or lossless pixels.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from mivln import video_review as review_model
from mivln.visual_messages import parse_action
from scripts import benchmark_navigation as source
from scripts import hssd_visual_dataset as batch
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, sha256, within
from scripts.language_annotations import read_json
from scripts.public_inputs import write_json, write_lines
from scripts.visual_inputs import validate_files


KIND = "reviewable_v_video_export_v1"
CODE = ("scripts/export_visual_videos.py", "mivln/video_review.py", "mivln/language_annotations.py", "scripts/visual_inputs.py",
        "mivln/visual_messages.py", "mivln/visual_frames.py")


def bindings():
    return {name: sha256(PROJECT_ROOT / name) for name in CODE}


def executable(name, explicit=None):
    """Find conda-local tools even when Python was invoked by absolute path."""
    found = shutil.which(str(explicit) if explicit else name)
    candidate = Path(found) if found else (Path(explicit) if explicit else Path(sys.executable).parent / name)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ValueError(f"{name} is required; activate the Habitat environment or pass --{name}")
    return str(candidate.resolve())


def roots_for(run_dir, prepared):
    return {"batch_run": run_dir, "batch_preparation": Path(read_json(run_dir / "manifest.json")["prepared_dir"]),
        "source_preparation": Path(prepared["source_dir"]),
        **{key+"_"+task["task_id"]: Path(task["reuse"][key]) for task in prepared["tasks"]
           if task["reuse"] for key in ("run_dir", "prepared_dir")}}


def verified_batch(run_dir, audit_dir):
    """Use the original completed audit only while all audited bytes still match.

    This is audit reuse, not a new native audit. Snapshot equality covers all
    audited raw evidence; the standard-library source validator also checks
    live scene/catalog/language dependencies outside those directory trees.
    """
    run_dir, audit_dir = run_dir.resolve(), audit_dir.resolve()
    run = read_json(run_dir / "manifest.json")
    if run.get("kind") != batch.RUN_KIND or run.get("code_bindings") != batch.code_bindings():
        raise ValueError("Batch run/execution code binding changed")
    prepared_dir = Path(run["prepared_dir"])
    if sha256(prepared_dir / "manifest.json") != run["prepared_manifest_sha256"]:
        raise ValueError("Batch preparation changed")
    prepared = read_json(prepared_dir / "manifest.json")
    if prepared.get("kind") != batch.KIND or prepared["code_bindings"] != batch.code_bindings():
        raise ValueError("Prepared batch kind/code changed")
    source.load_preparation(Path(prepared["source_dir"]))
    report = read_json(audit_dir / "audit.json")
    if (report.get("passed") is not True or report.get("source_files_unchanged") is not True
            or report.get("results_provisional") is not False
            or report.get("accepted_episode_slots") != prepared["planned_episodes"]
            or not report.get("episodes") or len(report["episodes"]) != prepared["planned_episodes"]):
        raise ValueError("Export requires a complete accepted read-only batch audit")
    cached = read_json(run_dir / "summary.json")
    if report != {**cached, "source_files_unchanged": True}:
        raise ValueError("Audit does not match the bound batch report")
    roots = roots_for(run_dir, prepared)
    old = read_json(audit_dir / "source_bindings.json")
    current = snapshot(roots)
    if old["before"] != old["after"] or current != old["after"]:
        raise ValueError("Audited source bytes/file roster changed")
    if [{key: row[key] for key in ("task_id", "case_id", "task_type", "condition")} for row in report["episodes"]] != prepared["episode_plan"]:
        raise ValueError("Audit task/condition roster differs from preparation")
    return prepared, report, roots, current


def select_rows(rows, selection):
    """A fixed eight-video QA set, separate from the full evaluation denominator."""
    if not rows or selection not in ("representative", "all"):
        raise ValueError("Choose a known, nonempty video selection")
    keys = [(row["task_id"], row["task_type"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source video identity")
    if selection == "all":
        return rows[:]
    wanted = [("task_001", family, condition) for family in source.navigation.TASK_TYPES for condition in ("oracle", "ignore")]
    wanted += [("task_007", "subgoal", condition) for condition in ("oracle", "ignore")]
    index = dict(zip(keys, rows))
    if any(key not in index for key in wanted):
        raise ValueError("The representative selection requires task_001 and task_007; use --selection all for another roster")
    return [index[key] for key in wanted]


def frame_plan(run_dir, prepared, row):
    """One frame per actual pre-action request, never alphabetical asset order.

    The last frame precedes terminal STOP at the same physical pose. STOP has no
    newly captured post-action image; no duplicate/new terminal observation is
    fabricated. Each selected frame remains linked to its actual bound action.
    """
    task = next(task for task in prepared["tasks"] if task["task_id"] == row["task_id"])
    task_run = Path(task["reuse"]["run_dir"]) if task["reuse"] else run_dir / "tasks" / row["task_id"] / "run"
    if Path(row["source_run"]).resolve() != task_run.resolve():
        raise ValueError("Selected source ownership changed")
    folder = task_run / "episodes" / row["case_id"] / row["condition"]
    public = folder / "public"
    validate_files(public / "requests.jsonl", public / "assets", public / "actions.jsonl")
    requests, actions = source.lines(public / "requests.jsonl"), source.lines(public / "actions.jsonl")
    records = source.lines(folder / "private/policy_records.jsonl")
    trajectory = read_json(folder / "private/trajectory.json")
    config = read_json(folder / "private/config.json")
    if len(requests) != len(actions) or len(records) != len(requests) or len(trajectory) != len(requests)+1:
        raise ValueError("Decision/action/trajectory counts disagree")
    if not requests or actions[-1]["action"] != "STOP":
        raise ValueError("Representative accepted episodes must end in actual STOP")
    frames = []
    for step, (request, action, record) in enumerate(zip(requests, actions, records)):
        if request != record["request"] or record["error"] or parse_action(record["raw_response"], request) != action:
            raise ValueError("Video action differs from preserved raw response")
        if request["step"] != step or trajectory[step+1]["action"] != action["action"]:
            raise ValueError("Video step differs from executed action")
        resource = request["observation"]["rgb"]
        image = within(public / "assets", resource["asset_id"] + ".png")
        if sha256(image) != resource["sha256"]:
            raise ValueError("Source PNG changed")
        frames.append({"frame_index": step, "observation_step": step, "request_id": request["request_id"],
            "asset_id": resource["asset_id"], "source_png": str(image), "source_sha256": resource["sha256"],
            "width": resource["width"], "height": resource["height"], "action": action["action"],
            "current_intervention": request["current_intervention"], "message_history": request["message_history"]})
    return folder, config, frames


def timestamp(milliseconds):
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def subtitles(frames, fps):
    """UTF-8 sidecar only: raw RGB pixels remain unannotated agent observations."""
    lines = []
    for index, frame in enumerate(frames):
        message = frame["current_intervention"]
        history = frame["message_history"]
        text = f"Step {frame['observation_step']} | next action: {frame['action']}"
        if message:
            text += "\nINTERVENTION: " + message["message"]
        elif history:
            text += "\nDelivered instruction: " + history[-1]["message"]
        lines += [str(index+1), f"{timestamp(round(index*1000/fps))} --> {timestamp(round((index+1)*1000/fps))}", text, ""]
    return "\n".join(lines)+"\n"


def probe_video(path, ffprobe):
    result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    if result.returncode:
        raise ValueError("FFprobe failed: " + result.stderr.strip())
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("Expected one video stream")
    return streams[0]


def check_media(metadata, video):
    try:
        actual_fps = float(Fraction(metadata["avg_frame_rate"]))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid encoded frame rate") from error
    if (metadata["codec_name"] != "h264" or int(metadata["nb_read_frames"]) != video["frame_count"]
            or int(metadata["width"]) != video["width"] or int(metadata["height"]) != video["height"]
            or not math.isclose(actual_fps, video["fps"], abs_tol=1e-5)
            or not math.isclose(float(metadata["duration"]), video["frame_count"]/video["fps"], abs_tol=max(.002, .01/video["fps"]))):
        raise ValueError("Encoded frame count, dimensions, fps or duration mismatch")


def encode(frames, destination, fps, ffmpeg):
    """Copy verified frames into an isolated numbered sequence; no source writes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v-video-", dir=destination.parent) as temporary:
        root = Path(temporary)
        for frame in frames:
            payload = Path(frame["source_png"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != frame["source_sha256"]:
                raise ValueError("Source changed during video encoding")
            (root / f"{frame['frame_index']:06d}.png").write_bytes(payload)
        encoded = root / "video.mp4"
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-framerate", str(fps),
            "-start_number", "0", "-i", str(root / "%06d.png"), "-frames:v", str(len(frames)),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
            capture_output=True, text=True)
        if result.returncode:
            raise ValueError("FFmpeg failed: " + result.stderr.strip())
        encoded.rename(destination)


def export(args):
    fps = args.fps
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("Use a finite display fps in (0, 120]")
    ffmpeg, ffprobe = executable("ffmpeg", args.ffmpeg), executable("ffprobe", args.ffprobe)
    prepared, report, roots, before = verified_batch(args.input_dir, args.audit_dir)
    roots["navigation_audit"] = args.audit_dir.resolve()
    before["navigation_audit"] = snapshot({"navigation_audit": args.audit_dir})["navigation_audit"]
    output = batch.disjoint_output(args.output_dir, list(roots.values()))
    selected = select_rows(report["episodes"], args.selection)
    plans = [(row, *frame_plan(args.input_dir.resolve(), prepared, row)) for row in selected]
    output.mkdir(parents=True)
    videos, failures = [], []
    for row, folder, config, frames in plans:
        identity = row["case_id"] + "_" + row["condition"]
        video = {"video_id": identity, **{key: row[key] for key in ("task_id", "task_type", "condition")},
            "expected_task_success": row["task_success"], "source_episode": str(folder.resolve()),
            "video": "videos/"+identity+".mp4", "subtitles": "videos/"+identity+".srt",
            "frames": "frames/"+identity+".jsonl", "frame_count": len(frames), "fps": fps,
            "width": frames[0]["width"], "height": frames[0]["height"],
            "intervention_step": config["events"][0]["trigger"]["value"],
            "initial_instruction": config["initial_instruction"], "intervention_message": config["events"][0]["utterance"]}
        try:
            if video["width"] % 2 or video["height"] % 2:
                raise ValueError("This yuv420p export requires even dimensions; do not silently pad source imagery")
            destination = output / video["video"]
            encode(frames, destination, fps, ffmpeg)
            check_media(probe_video(destination, ffprobe), video)
            (output / video["subtitles"]).write_text(subtitles(frames, fps), encoding="utf-8")
            write_lines(output / video["frames"], [{**frame, "video_time_seconds": frame["frame_index"]/fps} for frame in frames])
            for name in ("video", "subtitles", "frames"):
                video[name+"_sha256"] = sha256(output / video[name])
            videos.append(video)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"video_id": identity, "error": str(error)})
    unchanged = snapshot(roots) == before
    write_json(output / "export_report.json", {"planned_videos": len(selected), "exported_videos": len(videos),
        "planned_video_ids": [row["case_id"]+"_"+row["condition"] for row in selected],
        "exported_video_ids": [video["video_id"] for video in videos],
        "failures": failures, "source_files_unchanged": unchanged, "encoding_passed": not failures and unchanged})
    if failures or not unchanged:
        print("Video export incomplete; full planned count/failures retained, no reviewable manifest")
        return 1
    source_binding = {"roots": {key: str(path.resolve()) for key, path in roots.items()}, "snapshots": before}
    write_json(output / "private/source_bindings.json", source_binding)
    manifest = {"schema_version": 1, "kind": KIND, "export_id": uuid.uuid4().hex,
        "selection": args.selection, "fps": fps, "source_run": str(args.input_dir.resolve()),
        "source_audit": str(args.audit_dir.resolve()), "code_bindings": bindings(),
        "source_bindings_sha256": sha256(output / "private/source_bindings.json"), "videos": videos,
        "navigation_planned_episodes": report["planned_episodes"], "human_review_status": "pending"}
    write_json(output / "manifest.json", manifest)
    with (output / "review_sheet.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_model.COLUMNS)
        writer.writeheader()
        writer.writerows(review_model.worksheet(videos))
    index = ["# V 视频人工播放审核", "", "编码检查通过；以下全部待人工播放。Ignore 的批准表示忠实展示预期失败，不修改导航真值。", "",
        "视频每帧是该动作前观测，SRT 标注将执行动作；最后帧为 STOP 前相同位姿。fps 为播放速度，不是推理耗时。", "",
        "用支持 SRT 的播放器打开同名 MP4/SRT；字幕未烧录到原 RGB。", ""]
    for video in videos:
        index += [f"## {video['video_id']}", "", f"[视频]({video['video']}) · [字幕]({video['subtitles']}) · [逐帧索引]({video['frames']})", "",
            f"初始指令：{video['initial_instruction']}", "", f"介入：{video['intervention_message']}", "",
            f"投递步 {video['intervention_step']}，视频时间 {video['intervention_step']/fps:.2f}s；任务成功标注：{video['expected_task_success']}。", ""]
    (output / "review_index.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Exported {len(videos)} videos with source frame indexes; all playback reviews pending")
    return 0


def load_export(directory, ffprobe=None):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("kind") != KIND or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("code_bindings") != bindings()
            or manifest.get("human_review_status") != "pending"):
        raise ValueError("Video export/schema/code binding changed")
    path = directory / "private/source_bindings.json"
    if sha256(path) != manifest["source_bindings_sha256"]:
        raise ValueError("Video provenance seal changed")
    provenance = read_json(path)
    prepared, report, roots, current = verified_batch(Path(manifest["source_run"]), Path(manifest["source_audit"]))
    roots["navigation_audit"] = Path(manifest["source_audit"])
    current["navigation_audit"] = snapshot({"navigation_audit": roots["navigation_audit"]})["navigation_audit"]
    if ({key: str(value.resolve()) for key, value in roots.items()} != provenance["roots"]
            or current != provenance["snapshots"]):
        raise ValueError("Original audited video source changed")
    selected = select_rows(report["episodes"], manifest["selection"])
    if [video["video_id"] for video in manifest["videos"]] != [row["case_id"]+"_"+row["condition"] for row in selected]:
        raise ValueError("Export video roster changed")
    for video, row in zip(manifest["videos"], selected):
        _, config, frames = frame_plan(Path(manifest["source_run"]), prepared, row)
        if (any(video[key] != row[key] for key in ("task_id", "task_type", "condition"))
                or video["expected_task_success"] != row["task_success"] or video["frame_count"] != len(frames)
                or video["fps"] != manifest["fps"] or video["intervention_step"] != config["events"][0]["trigger"]["value"]
                or (video["width"], video["height"]) != (frames[0]["width"], frames[0]["height"])
                or video["initial_instruction"] != config["initial_instruction"]
                or video["intervention_message"] != config["events"][0]["utterance"]):
            raise ValueError("Video metadata differs from audited source")
        for name in ("video", "subtitles", "frames"):
            if sha256(within(directory, video[name])) != video[name+"_sha256"]:
                raise ValueError("Exported MP4/subtitles/frame index changed")
        expected_frames = [{**frame, "video_time_seconds": frame["frame_index"]/video["fps"]} for frame in frames]
        if source.lines(within(directory, video["frames"])) != expected_frames:
            raise ValueError("Exported frame order/provenance differs from actual requests")
        if within(directory, video["subtitles"]).read_text(encoding="utf-8") != subtitles(frames, video["fps"]):
            raise ValueError("Subtitle timing/message differs from actual delivery")
        if ffprobe:
            check_media(probe_video(within(directory, video["video"]), ffprobe), video)
    return manifest


def review(args):
    manifest = load_export(args.input_dir)
    with args.sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != review_model.COLUMNS:
            raise ValueError("Use the generated playback review header")
        decisions = review_model.import_sheet(manifest["videos"], list(reader))
    roots = read_json(args.input_dir / "private/source_bindings.json")["roots"]
    output = batch.disjoint_output(args.output_dir, [args.input_dir, *[Path(path) for path in roots.values()]])
    output.mkdir(parents=True)
    write_json(output / "review.json", {"kind": "human_v_video_review_v1", "export_id": manifest["export_id"],
        "export_manifest": str((args.input_dir / "manifest.json").resolve()), "export_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "recorded_at": datetime.now(timezone.utc).isoformat(), "reviewers_are_claimed_identities": True,
        "decisions": decisions})
    print(f"Saved {sum(row['decision'] != 'pending' for row in decisions)} explicit playback decisions; source export unchanged")
    return 0


def status(manifest, directory, review_dir=None):
    decisions = review_model.worksheet(manifest["videos"])
    if review_dir:
        record = read_json(review_dir / "review.json")
        if (record.get("kind") != "human_v_video_review_v1" or record["export_id"] != manifest["export_id"]
                or record["export_manifest_sha256"] != sha256(directory / "manifest.json")
                or Path(record["export_manifest"]).resolve() != (directory / "manifest.json").resolve()):
            raise ValueError("Playback review belongs to another or changed export")
        decisions = review_model.import_sheet(manifest["videos"], record["decisions"])
    counts = {key: sum(row["decision"] == key for row in decisions) for key in ("pending", "approve", "reject", "needs_revision")}
    return {"videos": len(decisions), "encoding_passed": True, "human_review": counts,
            "all_playback_approved": bool(decisions) and counts["approve"] == len(decisions), "navigation_labels_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "validate", "review", "status"):
        child = sub.add_parser(command)
        child.add_argument("--input-dir", type=Path, required=True)
        if command in ("export", "review"):
            child.add_argument("--output-dir", type=Path, required=True)
        if command == "export":
            child.add_argument("--audit-dir", type=Path, required=True)
            child.add_argument("--selection", choices=("representative", "all"), default="representative")
            child.add_argument("--fps", type=float, default=10.)
            child.add_argument("--ffmpeg", type=Path)
            child.add_argument("--ffprobe", type=Path)
        if command == "validate":
            child.add_argument("--ffprobe", type=Path)
        if command == "review":
            child.add_argument("--sheet", type=Path, required=True)
        if command in ("status", "validate"):
            child.add_argument("--review-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command in ("export", "review"):
            return globals()[args.command](args)
        manifest = load_export(args.input_dir, executable("ffprobe", args.ffprobe) if args.command == "validate" else None)
        print(json.dumps(status(manifest, args.input_dir, args.review_dir), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, csv.Error) as error:
        print(f"V video operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
