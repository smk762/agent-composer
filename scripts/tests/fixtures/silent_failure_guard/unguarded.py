"""Silent-failure guard fixture: code the rule must flag (scripts/tests/test_silent_failure_guard.py). Never imported."""

import json


def pair(labels, scores):
    return list(zip(labels, scores))  # a short list silently drops the rest


def load(paths, read):
    out = []
    for p in paths:
        try:
            out.append(read(p))
        except Exception:
            continue  # the failure vanishes
    return out


def probe(fn):
    try:
        return fn()
    except:  # noqa: E722
        pass


def unexplained(fn):
    try:
        return fn()
    except BaseException:  # noqa: S110
        pass


def load_state(path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}  # a corrupt state file becomes a plausible empty state


def notify(send, msg) -> None:
    try:
        send(msg)
    except BaseException:
        return  # a bare return swallows it like pass


def ellipsis(fn):
    try:
        return fn()
    except Exception:
        ...


def dead_store(fn):
    try:
        return fn()
    except Exception as exc:
        _ = exc


def explained_to_nobody(fn):
    try:
        return fn()
    except Exception:
        "best effort"
        pass


def reset_then_use(path, read_json):
    try:
        state = read_json(path)
    except Exception:
        state = {}
    return state


def suppressed(fn, contextlib):
    with contextlib.suppress(Exception):
        fn()
