"""Policy adapter interface. Policies see only observations (except the oracle)."""

from __future__ import annotations


class Policy:
    name = "base"
    privileged = False  # True only for the oracle: the runner hands it truth

    def reset(self, obs: dict, truth: dict | None = None) -> None:
        self.episode_id = obs["manifest"]["episode_id"]
        self.base_currency = obs["manifest"]["base_currency"]

    def act(self, obs: dict) -> dict:
        raise NotImplementedError


def submit(episode_id: str, findings: list[dict]) -> dict:
    return {"tool": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings}}}


class Pager:
    """Helper that walks every page of a table through list_records."""

    def __init__(self, table: str, flt: dict | None = None, max_pages: int = 100) -> None:
        self.table = table
        self.flt = flt
        self.cursor = None
        self.records: list[dict] = []
        self.done = False
        self.pages = 0
        self.max_pages = max_pages

    def action(self) -> dict:
        args = {"table": self.table, "cursor": self.cursor, "limit": 50}
        if self.flt:
            args["filter"] = self.flt
        return {"tool": "list_records", "args": args}

    def feed(self, obs: dict) -> None:
        self.pages += 1
        if not obs.get("ok"):
            self.done = True
            return
        res = obs["result"]
        self.records.extend(res["records"])
        self.cursor = res["next_cursor"]
        if self.cursor is None or self.pages >= self.max_pages:
            self.done = True
