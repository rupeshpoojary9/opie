from conftest import FakeExtractor

from opie.llm.offline import OfflineExtractor
from opie.pipeline.graph import Pipeline

_LABEL_TEXT = """
NUTRITION per 100g
Energy 512 kcal
Fat 30 g
of which saturates 12 g
Carbohydrate 55 g
of which sugars 34 g
Fibre 3 g
Protein 8 g
Salt 1.2 g

Ingredients: wheat flour, sugar, palm oil, soy lecithin, glucose-fructose syrup, salt.

HIGH IN PROTEIN. No added sugar.
"""


def test_offline_extractor_parses_nutrition():
    ex = OfflineExtractor()
    panel = ex.extract_nutrition(b"", _LABEL_TEXT)
    assert panel.energy_kcal_100g.value == 512
    assert panel.sugars_100g.value == 34
    assert panel.saturated_fat_100g.value == 12
    assert panel.salt_100g.value == 1.2
    # every parsed field carries evidence
    assert panel.sugars_100g.evidence is not None


def test_offline_extractor_parses_ingredients():
    ex = OfflineExtractor()
    ings = ex.extract_ingredients(b"", _LABEL_TEXT)
    names = [i.name for i in ings]
    assert "wheat flour" in names
    assert "soy lecithin" in names


def test_offline_extractor_parses_claims():
    ex = OfflineExtractor()
    claims = ex.extract_claims(b"", _LABEL_TEXT)
    normalized = {c.normalized for c in claims}
    assert "high_protein" in normalized
    assert "no_added_sugar" in normalized


def test_pipeline_end_to_end_with_fake_extractor():
    ex = FakeExtractor(
        nutrition={"energy_kcal_100g": 512, "sugars_100g": 34, "saturated_fat_100g": 12,
                   "fat_100g": 30, "carbohydrates_100g": 55, "salt_100g": 1.2,
                   "proteins_100g": 8, "fiber_100g": 3},
        ingredients=["wheat flour", "sugar", "soy lecithin", "glucose-fructose syrup"],
        claims=["high in protein"],
    )
    pipe = Pipeline(extractor=ex)
    result = pipe.run("demo", b"fake-image", profiles=["general", "diabetic"])

    # extraction + taxonomy annotation happened
    assert result.nutrition.sugars_100g.value == 34
    by_name = {i.name: i for i in result.ingredients}
    assert by_name["soy lecithin"].enumber == "E322"
    assert by_name["glucose-fructose syrup"].ultra_processing_marker is True

    # validation + scoring populated
    assert result.score.base_grade in {"A", "B", "C", "D", "E"}
    assert len(result.score.personalized) == 2
    assert result.latency_ms is not None
    assert result.backend == "fake"


def test_pipeline_engine_reported():
    pipe = Pipeline(extractor=FakeExtractor())
    assert pipe.engine in {"langgraph", "sequential"}
