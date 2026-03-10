import numpy as np
import pytest

from ai_engine.environment.reward import RewardFunction

PRICING = {"us-east-1": {"m5.large": 0.096, "on_demand_per_vcpu_hr": 0.048}}


def _state(sla_ms=200.0, priority=2):
    s = np.zeros(45, dtype=np.float32)
    s[6] = priority / 4.0
    s[7] = sla_ms / 1000.0
    s[30] = 45.0 / 1000.0   # us-east-1 latency
    s[20] = 415.0 / 600.0   # us-east-1 carbon
    return s


def _action(**kwargs):
    base = {
        "cloud": "aws", "region": "us-east-1",
        "generic_region": "us-east-1", "instance_type": "m5.large",
        "scaling_level": 1, "purchase_option": "on_demand",
        "sla_tier": 2, "requires_migration": False,
    }
    return {**base, **kwargs}


@pytest.fixture
def rf():
    return RewardFunction({})


def test_reward_in_bounds(rf):
    r, _ = rf.compute(_action(), _state(), PRICING)
    assert -10.0 <= r <= 10.0


def test_spot_better_than_on_demand(rf):
    r_od, _  = rf.compute(_action(purchase_option="on_demand"), _state(), PRICING)
    r_spot, _ = rf.compute(_action(purchase_option="spot"),     _state(), PRICING)
    assert r_spot > r_od


def test_migration_reduces_reward(rf):
    r_local, _ = rf.compute(_action(requires_migration=False), _state(), PRICING)
    r_migr, _  = rf.compute(_action(requires_migration=True),  _state(), PRICING)
    assert r_local > r_migr


def test_components_present(rf):
    _, comps = rf.compute(_action(), _state(), PRICING)
    for key in ("cost", "latency", "carbon", "sla", "migration", "total"):
        assert key in comps


def test_low_carbon_region_better(rf):
    """us-west-2 (192 gCO2) should yield higher carbon reward than ap-northeast-1 (506)."""
    s_green = _state()
    s_green[21] = 192.0 / 600.0   # us-west-2 slot
    s_dirty = _state()
    s_dirty[21] = 506.0 / 600.0

    a_green = _action(generic_region="us-west-2")
    a_dirty = _action(generic_region="ap-northeast-1")

    r_green, _ = rf.compute(a_green, s_green, PRICING)
    r_dirty, _ = rf.compute(a_dirty, s_dirty, PRICING)
    assert r_green >= r_dirty