from polymarket_pipeline.filters import filter_scheduled_markets


def test_filter_scheduled_markets_basic():
    markets = [
        {"id": "1", "question": "A", "startDate": "2026-01-01T12:00:00Z", "clobTokenIds": "[\"tok1\",\"tok2\"]", "outcomes": ["Yes", "No"]},
        {"id": "2", "question": "B"},
    ]
    out = filter_scheduled_markets(markets)
    assert len(out) == 1
    assert out[0]["id"] == "1"
    assert out[0]["_tokens"][0]["token_id"] == "tok1"
