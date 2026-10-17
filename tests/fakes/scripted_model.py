"""Model double for task tests: records ``generate_response`` calls, returns scripts."""

from __future__ import annotations

from typing import Any

from tests.fakes.recorder import RECORDER, RESPONDER, digest, snapshot


class ScriptedModel:
    """Stand-in for a model wrapper with the ``generate_response`` contract.

    Long, repeated arguments (system prompts and few-shot examples) are recorded as
    digests; their full contents are kept once in :attr:`contents_by_digest`.

    Attributes:
        pretrained: Model name, read by the tasks that derive ``model_name`` from it.
        contents_by_digest: Full system prompts and few-shot data keyed by digest.
    """

    def __init__(self, pretrained: str = "org/scripted-model") -> None:
        """Create a model double reporting ``pretrained`` as its name."""
        self.pretrained = pretrained
        self.contents_by_digest: dict[str, Any] = {}

    def _remember(self, value: Any) -> str | None:
        """Store ``value`` under its digest and return the digest (``None`` stays ``None``)."""
        if value is None:
            return None
        key = digest(value)
        self.contents_by_digest[key] = snapshot(value)
        return key

    def generate_response(
        self,
        text: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        few_shot_data: list[dict[str, Any]] | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Record the request and return the next scripted response."""
        RECORDER.record(
            "generate_response",
            text=text,
            images=images,
            system_prompt=self._remember(system_prompt),
            few_shot_data=self._remember(few_shot_data),
            gen_kwargs=gen_kwargs,
        )
        return RESPONDER.next()
