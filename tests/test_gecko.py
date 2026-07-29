import pytest
from scouter.gecko import parse_pools, age_days, effective_mcap

PAYLOAD = {
    "data": [{
        "id": "solana_POOL1",
        "type": "pool",
        "attributes": {
            "address": "POOL1", "name": "WCC / USDC",
            "pool_created_at": "2026-07-29T17:57:15Z",
            "base_token_price_usd": "0.0000048113",
            "fdv_usd": "4426.942206", "market_cap_usd": None,
            "reserve_in_usd": "5200.5",
            "volume_usd": {"h1": "120.0", "h24": "3400.0"},
            "transactions": {"h24": {"buys": 40, "sells": 25}},
        },
        "relationships": {
            "base_token": {"data": {"id": "solana_MINTWCC", "type": "token"}},
            "dex": {"data": {"id": "raydium", "type": "dex"}},
        },
    }],
    "included": [
        {"id": "solana_MINTWCC", "type": "token",
         "attributes": {"address": "MINTWCC", "symbol": "WCC",
                        "name": "World Cup Coin", "decimals": 6}},
        {"id": "raydium", "type": "dex", "attributes": {"name": "Raydium"}},
    ],
}


def test_parses_core_fields():
    p = parse_pools(PAYLOAD)[0]
    assert p["address"] == "POOL1"
    assert p["base_token_address"] == "MINTWCC"
    assert p["base_token_symbol"] == "WCC"
    assert p["dex"] == "Raydium"
    assert p["fdv_usd"] == pytest.approx(4426.942206)
    assert p["reserve_usd"] == pytest.approx(5200.5)
    assert p["txns_h24"] == 65


def test_missing_reserve_is_none_not_zero():
    """Live API returns pools with $7M volume and no reserve field. Treating
    that as 0 would silently fail the liquidity floor."""
    import copy
    bad = copy.deepcopy(PAYLOAD)
    del bad["data"][0]["attributes"]["reserve_in_usd"]
    assert parse_pools(bad)[0]["reserve_usd"] is None

    zero = copy.deepcopy(PAYLOAD)
    zero["data"][0]["attributes"]["reserve_in_usd"] = "0"
    assert parse_pools(zero)[0]["reserve_usd"] == 0.0


def test_effective_mcap_falls_back_to_fdv():
    assert effective_mcap({"market_cap_usd": None, "fdv_usd": 4426.9}) == 4426.9
    assert effective_mcap({"market_cap_usd": 900.0, "fdv_usd": 4426.9}) == 900.0


def test_base_token_address_from_relationship_id_when_not_included():
    import copy
    bare = copy.deepcopy(PAYLOAD)
    bare["included"] = []
    assert parse_pools(bare)[0]["base_token_address"] == "MINTWCC"


def test_age_days():
    assert age_days(None) is None
    assert age_days("2020-01-01T00:00:00Z") > 2000


def test_empty_payload():
    assert parse_pools(None) == []
    assert parse_pools({"data": []}) == []
