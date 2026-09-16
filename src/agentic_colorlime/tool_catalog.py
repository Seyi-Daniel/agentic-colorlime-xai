from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCard:
    method: str
    function_name: str
    explainer: str = "lime"
    segmentation: str = ""

    @property
    def segmentation_method(self) -> str:
        return self.segmentation or self.method

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TOOL_CARDS: dict[str, ToolCard] = {
    "slic": ToolCard(
        method="slic",
        function_name="run_slic_lime",
    ),
    "quickshift": ToolCard(
        method="quickshift",
        function_name="run_quickshift_lime",
    ),
    "felzenszwalb": ToolCard(
        method="felzenszwalb",
        function_name="run_felzenszwalb_lime",
    ),
    "watershed": ToolCard(
        method="watershed",
        function_name="run_watershed_lime",
    ),
    "colorlime": ToolCard(
        method="colorlime",
        function_name="run_colorlime",
    ),
}

EXPLAINERS = {
    "lime": "Image LIME with its default weighted Ridge surrogate.",
    "lime_lasso": "Image LIME with a Lasso surrogate and configured positive alpha; a sparse LIME variant, not LEMON.",
    "shap": "Kernel SHAP over segment-presence features, using an all-hidden background and probability units. At most 10 nonzero features are fitted; inspect diagnostics and segment count.",
}

for _family in ("lime_lasso", "shap"):
    for _segmentation in tuple(TOOL_CARDS):
        if TOOL_CARDS[_segmentation].explainer != "lime":
            continue
        _method = f"{_family}_{_segmentation}"
        TOOL_CARDS[_method] = ToolCard(
            method=_method, function_name=f"run_{_method}",
            explainer=_family, segmentation=_segmentation,
        )


class ToolCatalog:
    """Identifier catalogue for every registered explanation tool.

    There is deliberately no shortlist or retrieval policy in this version.
    Every unattempted registered tool is exposed to the agent at each decision
    point. Tool behaviour is available through local source inspection rather
    than hand-written strengths, limitations, or recommendations.
    """

    def __init__(self, cards: dict[str, ToolCard] | None = None) -> None:
        self.cards = dict(cards or TOOL_CARDS)

    def all_cards(self) -> list[ToolCard]:
        return [self.cards[name] for name in sorted(self.cards)]

    def get(self, method: str) -> ToolCard:
        return self.cards[method]

    def cards_for_methods(self, methods: list[str] | set[str]) -> list[ToolCard]:
        return [self.cards[name] for name in sorted(methods)]
