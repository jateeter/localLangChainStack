"""Runtime-neutral agent behaviors, rendered for A2A (the template level of #3).

A behavior is RealityEngine_Machines' runtime-neutral agent description
(`schemas/agent-behavior.schema.json`, decision GOV-SD-08): role, objective,
instructions, prohibitions, an output contract and the response mapping that
names its completion fields. It names no runtime. This module renders one for
this service:

* **Discovery:** each behavior is an A2A skill on the Agent Card, with the
  behavior id as the skill id (RealityEngine_CI docs/A2A_FRAMEWORK_SPEC.md 4.2:
  a binding names an agent and a skill id).
* **Dispatch:** a request whose RealityEngine extension names `dispatch.skill`
  runs that behavior. Its instructions and output contract are the system turn,
  and the step's text is the user turn.
* **Completion:** the reply's `<field>: <value>` lines are parsed against the
  behavior's `responseMapping`. They are returned as a `completion` artifact
  (JSON), beside the text, for the PE's commit gate. Nothing is written back
  from here: the commit gate is in-process on the engine side (spec 5.3).

Two roles are refused, never imitated:

* `deterministic-tool`: its result must come from a tool. This service has
  none, so a model must not stand in for one.
* any behavior whose `responseMapping` names no field: there would be nothing
  for the commit gate to read.

Capabilities (`frame.read`, `pod.write`, ...) are declared, not granted: this
service holds no lifecycle credential and exposes no tools. Frame-scoped tools
are #5. The per-clone lifecycle (create, version, delete) is the rest of #3.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REFUSED_ROLES = {"deterministic-tool": "its result must come from a tool, and this service has none"}


@dataclass(frozen=True)
class Behavior:
    behavior_id: str
    display_name: str
    role: str
    domain: str
    objective: str
    instructions: str
    output_contract: str
    fields: tuple[tuple[str, str], ...]  # (response key, value type)
    workflow: str | None
    capabilities: tuple[str, ...]

    @property
    def refusal(self) -> str | None:
        if self.role in REFUSED_ROLES:
            return f"behavior {self.behavior_id} ({self.role}) is not run here: {REFUSED_ROLES[self.role]}"
        if not self.fields:
            return f"behavior {self.behavior_id} names no completion field"
        return None

    def system_prompt(self) -> str:
        return f"{self.instructions}\n\nOUTPUT CONTRACT\n{self.output_contract}"

    def parse_completion(self, text: str) -> dict:
        """The completion fields the reply states, by the output contract's `<field>: <value>` pairs.

        A pair may stand on its own line or share one (the universe model often
        writes them on one line).

        A field the reply does not state is null; it is never guessed. A binary
        field takes only 0 or 1; a continuous one a number.
        """
        values: dict[str, object] = {}
        for key, kind in self.fields:
            pattern = r"(0|1)" if kind == "binary" else r"(-?\d+(?:\.\d+)?)"
            m = re.search(rf"(?i)(?:^|[\s,;`*_-]){re.escape(key)}[\s`*_]*:\s*{pattern}\b", text)
            if m is None:
                values[key] = None
            else:
                values[key] = int(m.group(1)) if kind == "binary" else float(m.group(1))
        m = re.search(r"(?im)(?:^|[\s,;`*_-])rationale[\s`*_]*:\s*(.+)$", text)
        return {"fields": values, "rationale": m.group(1).strip() if m else None,
                "complete": all(v is not None for v in values.values())}


def from_document(doc: dict) -> Behavior:
    if not str(doc.get("schemaVersion", "")).startswith("1."):
        raise ValueError(f"unsupported behavior schemaVersion {doc.get('schemaVersion')!r}")
    reasoning = doc["reasoning"]
    fields = tuple(
        ((f.get("extract") or {}).get("responseKey") or f["semantic"], f.get("valueType", "binary"))
        for f in (doc.get("responseMapping") or {}).get("fields", [])
    )
    return Behavior(
        behavior_id=doc["behaviorId"],
        display_name=doc.get("displayName") or doc["behaviorId"],
        role=doc["role"],
        domain=doc.get("domain", ""),
        objective=reasoning["objective"],
        instructions=reasoning["instructions"],
        output_contract=reasoning["outputContract"],
        fields=fields,
        workflow=(doc.get("dispatch") or {}).get("workflow"),
        capabilities=tuple(doc.get("capabilities", [])),
    )


def load_behaviors(directory: str | Path | None) -> dict[str, Behavior]:
    """Every `*.agent.json` behavior in `directory`, by behavior id. None or absent: no behaviors."""
    if not directory:
        return {}
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"behaviors directory not found: {root}")
    out: dict[str, Behavior] = {}
    for path in sorted(root.glob("*.agent.json")):
        behavior = from_document(json.loads(path.read_text(encoding="utf-8")))
        if behavior.behavior_id in out:
            raise ValueError(f"duplicate behavior id {behavior.behavior_id} ({path.name})")
        out[behavior.behavior_id] = behavior
    logger.info("loaded %d behaviors from %s", len(out), root)
    return out
