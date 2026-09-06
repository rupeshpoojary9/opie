from opie.schemas import NutritionPanel, ProductIntelligence


def test_nutrition_panel_field_keys():
    keys = set(NutritionPanel().as_dict().keys())
    assert "energy_kcal_100g" in keys
    assert "sugars_100g" in keys
    assert len(keys) == 9


def test_product_intelligence_roundtrip():
    p = ProductIntelligence(product_id="123")
    dumped = p.model_dump_json()
    back = ProductIntelligence.model_validate_json(dumped)
    assert back.product_id == "123"
    assert back.validation.passed is True
    assert back.score.personalized == []
