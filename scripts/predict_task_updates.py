"""Generate raw predictions from a Chat Completions compatible model service.

Only frozen public requests, their hashes and instructions are read. Evaluator
references and the source dataset are deliberately unnecessary for inference.
Each request starts a separate conversation, preserving the three context modes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from mivln.task_update import strict_json, validate_request
from mivln.task_update_prompts import (PROTOCOL as PROMPT_PROTOCOL, VARIANTS, check_variant,
                                      fingerprint, system_prompt as render_system, user_prompt)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def read_json(path):
    return strict_json(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_public_requests(folder):
    """Check the frozen input without opening manifest['dataset_path']."""
    manifest = read_json(folder / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "conditional_task_update_requests":
        raise ValueError("Unsupported public request export")
    for name, key in (("requests.jsonl", "requests_sha256"),
                      ("request_index.json", "request_index_sha256")):
        if sha256(folder / name) != manifest[key]:
            raise ValueError(f"Frozen public input changed: {name}")
    index = read_json(folder / "request_index.json")
    identities = manifest["request_ids"]
    if (not isinstance(index, dict) or list(index) != identities
            or manifest["planned_records"] != len(index)):
        raise ValueError("Public request roster changed")
    requests = {}
    for line in (folder / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        request = validate_request(strict_json(line))
        identity = request["request_id"]
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if identity in requests or index.get(identity) != digest:
            raise ValueError("Unknown, duplicate or mismatched public request")
        requests[identity] = request
    if list(requests) != [identity for identity in index if index[identity] is not None]:
        raise ValueError("Public request list is incomplete or reordered")
    missing_context = {identity for identity in index if index[identity] is None}
    if set(manifest["context_errors"]) != missing_context:
        raise ValueError("Context failures disagree with the planned roster")
    instructions = (folder / "instructions.txt").read_text(encoding="utf-8")
    if not instructions.strip():
        raise ValueError("Instructions must not be empty")
    return manifest, index, requests, instructions


def endpoint_for(base_url):
    """The supplied base includes its API prefix, usually /v1."""
    parts = urlsplit(base_url)
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError("Use an HTTP(S) API base URL without credentials, query or fragment")
    return base_url.rstrip("/") + "/chat/completions"


def payload_for(model, system_prompt, request, temperature, max_tokens,
                thinking_mode="default", prompt_variant="baseline"):
    """One body builder shared by the frozen preview and actual HTTP call."""
    payload = {"model": model, "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt(request, prompt_variant)}],
        "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    if thinking_mode != "default":
        # vLLM/Qwen apply this boolean when rendering the model's chat template.
        # Keep it out of generic requests unless explicitly selected: other
        # providers may use a different parameter or reject this extension.
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode == "on"}
    return payload


def complete(endpoint, model, system_prompt, request, api_key, timeout, temperature, max_tokens,
             thinking_mode="default", prompt_variant="baseline"):
    """One stateless HTTP call; preserve model text without repairing its JSON."""
    payload = payload_for(model, system_prompt, request, temperature, max_tokens, thinking_mode, prompt_variant)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    http_request = Request(endpoint, data=encode(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(http_request, timeout=timeout) as response:
        result = strict_json(response.read().decode("utf-8"))
    choice = result["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model service did not return nonempty message content")
    # A length-limited response remains raw. The scorer must see any truncation
    # or Markdown fences rather than silently benefiting from answer repair.
    return content, {"finish_reason": choice.get("finish_reason"),
                     "reported_model": result.get("model"), "usage": result.get("usage")}


def generate(args):
    """Retain every planned slot, including failed calls and interruption tails."""
    thinking_mode = getattr(args, "thinking_mode", "default")
    prompt_variant = getattr(args, "prompt_variant", "baseline")
    prepare_only = getattr(args, "prepare_only", False)
    check_variant(prompt_variant)
    if (not args.model.strip() or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.temperature) or args.temperature < 0 or args.max_tokens <= 0
            or thinking_mode not in ("default", "off", "on")):
        raise ValueError("Model and generation parameters are invalid")
    endpoint = endpoint_for(args.base_url)
    folder = args.requests_dir.resolve()
    manifest, index, requests, instructions = load_public_requests(folder)
    system_prompt = render_system(instructions, prompt_variant)
    api_key = os.environ.get(args.api_key_env, "")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = {"kind": "chat_task_update_prompt_plan" if prepare_only else "chat_task_update_predictions",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requests_dir": str(folder), "requests_sha256": manifest["requests_sha256"],
        "request_index_sha256": manifest["request_index_sha256"],
        "instructions_sha256": sha256(folder / "instructions.txt"),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "model": args.model, "base_url": args.base_url, "temperature": args.temperature,
        "prompt_protocol": PROMPT_PROTOCOL, "prompt_variant": prompt_variant, "prepare_only": prepare_only,
        "thinking_mode": thinking_mode,
        "chat_template_kwargs": ({} if thinking_mode == "default" else {"enable_thinking": thinking_mode == "on"}),
        "max_tokens": args.max_tokens, "timeout_seconds": args.timeout, "planned_records": len(index),
        "attempted": 0, "received": 0, "failed": 0, "interrupted": False,
        "notes": "Raw outputs; no JSON repair, retries, reference answers or dialogue carry-over"}
    (output / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    # Canonical request hashes bind evaluation inputs; a separate file records
    # the ACTUAL presented message/body, since a timeline is not the source JSON.
    changed, ready = 0, 0
    baseline_system = render_system(instructions, "baseline")
    with (output / "model_inputs.jsonl").open("x", encoding="utf-8") as stream:
        for identity, digest in index.items():
            request = requests.get(identity)
            payload = None
            if request is not None:
                ready += 1
                payload = payload_for(args.model, system_prompt, request, args.temperature,
                                      args.max_tokens, thinking_mode, prompt_variant)
                baseline = payload_for(args.model, baseline_system, request, args.temperature,
                                       args.max_tokens, thinking_mode, "baseline")
                changed += payload != baseline
            stream.write(encode({"request_id": identity, "request_sha256": digest,
                "payload": payload, "payload_sha256": fingerprint(payload) if payload is not None else None}) + "\n")
    run.update(model_inputs_sha256=sha256(output / "model_inputs.jsonl"), ready_requests=ready,
               changed_request_count=changed, input_context_failures=len(index) - ready)
    if prepare_only:
        # No predictions or responses are fabricated for a CPU-only preview.
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {ready}/{len(index)} inputs for {prompt_variant}; changed: {changed}; no model calls")
        return 0 if ready == len(index) else 1
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
            (output / "responses.jsonl").open("x", encoding="utf-8") as responses:
        for position, (identity, digest) in enumerate(index.items(), 1):
            content, metadata, error = None, {}, None
            if run["interrupted"]:
                error = "Not attempted after interruption"
            elif identity not in requests:
                error = "Public context construction failed"
            else:
                run["attempted"] += 1
                try:
                    content, metadata = complete(endpoint, args.model, system_prompt, requests[identity],
                        api_key, args.timeout, args.temperature, args.max_tokens, thinking_mode, prompt_variant)
                except KeyboardInterrupt:
                    run["interrupted"] = True
                    error = "Interrupted during model call"
                except HTTPError as failure:
                    # Avoid logging authentication details or server error bodies.
                    error = f"HTTP error {failure.code}"
                    # HTTPError may have no response body (including Python 3.9
                    # transport stubs); closing its empty wrapper can itself fail.
                    if failure.fp is not None:
                        failure.close()
                except (URLError, OSError, ValueError, KeyError, IndexError, TypeError) as failure:
                    error = f"Model call failed: {type(failure).__name__}"
            run["received" if error is None else "failed"] += 1
            predictions.write(encode({"request_id": identity, "request_sha256": digest,
                                      "prediction": content}) + "\n")
            responses.write(encode({"request_id": identity, "error": error, **metadata}) + "\n")
            predictions.flush()
            responses.flush()
            print(f"[{position}/{len(index)}] {identity}: {'received' if error is None else error}", flush=True)
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    run["predictions_sha256"] = sha256(output / "predictions.jsonl")
    (output / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Predictions: {output / 'predictions.jsonl'}")
    return 1 if run["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="API base including /v1 when required")
    parser.add_argument("--model", required=True, help="Model identifier exposed by the service")
    parser.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking-mode", choices=("default", "off", "on"), default="default",
                        help="vLLM/Qwen chat-template extension; default leaves the service setting unchanged")
    parser.add_argument("--prompt-variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--prepare-only", action="store_true", help="CPU: freeze actual input bodies without calling a model")
    parser.add_argument("--output-dir", type=Path, required=True)
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
