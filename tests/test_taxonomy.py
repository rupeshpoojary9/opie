from opie.schemas import Ingredient
from opie.taxonomy.taxonomy import Taxonomy, normalize_ingredients


def _ing(name):
    return Ingredient(name=name, rank=1)


def test_additive_enumber():
    tax = Taxonomy()
    assert tax.additive("soy lecithin")[0] == "E322"
    assert tax.additive("citric acid")[0] == "E330"


def test_allergen_detection():
    tax = Taxonomy()
    assert tax.allergen("wheat flour") == "gluten"
    assert tax.allergen("whole milk") == "milk"
    assert tax.allergen("water") is None


def test_ultra_processing_marker():
    tax = Taxonomy()
    assert tax.is_ultra_processing("glucose-fructose syrup") is True
    assert tax.is_ultra_processing("carrot") is False


def test_classify():
    tax = Taxonomy()
    assert tax.classify("wheat flour") == "cereal"
    assert tax.classify("sunflower oil") == "oil_fat"


def test_annotate_full():
    ings = normalize_ingredients([_ing("wheat flour"), _ing("sugar"),
                                  _ing("soy lecithin"), _ing("glucose-fructose syrup")])
    by_name = {i.name: i for i in ings}
    assert by_name["wheat flour"].is_allergen is True
    assert by_name["wheat flour"].taxonomy == "cereal"
    assert by_name["soy lecithin"].enumber == "E322"
    assert by_name["glucose-fructose syrup"].ultra_processing_marker is True
