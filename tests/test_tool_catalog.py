from agentic_colorlime.segmentations import SEGMENTATION_FUNCTIONS
from agentic_colorlime.tool_catalog import ToolCatalog


def test_catalog_has_metadata_for_every_registered_method():
    catalog = ToolCatalog()
    assert set(catalog.cards) == set(SEGMENTATION_FUNCTIONS)
    for card in catalog.all_cards():
        assert set(card.to_dict()) == {"method", "function_name"}


def test_cards_for_methods_returns_every_requested_method():
    catalog = ToolCatalog()
    requested = {"slic", "colorlime", "watershed"}
    returned = {card.method for card in catalog.cards_for_methods(requested)}
    assert returned == requested
