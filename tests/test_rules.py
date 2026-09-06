from conftest import panel

from opie.rules.engine import RulesEngine
from opie.schemas import ProductIntelligence, Severity


def _validate(**fields):
    prod = ProductIntelligence(product_id="t", nutrition=panel(**fields))
    return RulesEngine().validate(prod)


def test_clean_product_passes():
    r = _validate(energy_kcal_100g=400, carbohydrates_100g=50, proteins_100g=10,
                  fat_100g=15, saturated_fat_100g=3, sugars_100g=10,
                  salt_100g=0.5, sodium_100g=0.2, fiber_100g=2)
    assert r.passed is True
    assert r.review_required is False


def test_sugars_gt_carbs_flagged():
    r = _validate(sugars_100g=50, carbohydrates_100g=10)
    assert any(f.rule == "sugars_le_carbs" for f in r.flags)
    assert r.passed is False


def test_saturated_gt_fat_flagged():
    r = _validate(fat_100g=5, saturated_fat_100g=20)
    assert any(f.rule == "saturated_le_fat" for f in r.flags)


def test_impossible_percent_flagged():
    r = _validate(fat_100g=150)
    assert any(f.rule == "percent_bounds" for f in r.flags)


def test_energy_macro_mismatch_flagged():
    # macros imply ~375 kcal; claim 1500
    r = _validate(energy_kcal_100g=1500, carbohydrates_100g=50, proteins_100g=10, fat_100g=15)
    assert any(f.rule == "energy_macro_balance" for f in r.flags)


def test_low_confidence_routes_to_review():
    prod = ProductIntelligence(product_id="t", nutrition=panel())
    prod.nutrition.sugars_100g.value = 10.0
    prod.nutrition.sugars_100g.confidence = 0.2   # below review threshold
    r = RulesEngine().validate(prod)
    assert r.review_required is True


def test_high_threshold_severity():
    r = _validate(sugars_100g=40)
    assert any(f.severity == Severity.high and f.rule == "sugars_high" for f in r.flags)
