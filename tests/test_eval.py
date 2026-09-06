from conftest import FakeExtractor, make_off_product

from opie.data.off import OFFProduct
from opie.eval.harness import run_eval
from opie.eval.metrics import PRF, cohen_kappa, ingredient_prf, numeric_match, nutrition_prf
from opie.pipeline.graph import Pipeline


def test_numeric_match_tolerance():
    assert numeric_match(100, 100) is True
    assert numeric_match(103, 100) is True          # within 5%
    assert numeric_match(120, 100) is False         # outside 5%
    assert numeric_match(0.3, 0.1) is True          # within abs floor
    assert numeric_match(None, 5) is False


def test_prf_math():
    prf = PRF(tp=8, fp=2, fn=2)
    assert prf.precision == 0.8
    assert prf.recall == 0.8
    assert round(prf.f1, 3) == 0.8


def test_nutrition_prf_hallucination_and_miss():
    pred = {"sugars_100g": 10.0, "fat_100g": 5.0, "salt_100g": None}
    gt = {"sugars_100g": 10.0, "fat_100g": None, "salt_100g": 1.0}
    prf = nutrition_prf(pred, gt)
    assert prf.tp == 1   # sugars correct
    assert prf.fp == 1   # fat hallucinated
    assert prf.fn == 1   # salt missed


def test_ingredient_prf():
    prf = ingredient_prf(["wheat flour", "sugar"], ["wheat flour", "salt"])
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 1


def test_cohen_kappa_perfect():
    labels = ["A", "B", "C", "D", "E"]
    assert cohen_kappa(["A", "B", "C"], ["A", "B", "C"], labels) == 1.0


def test_run_eval_end_to_end():
    gt_nut = {"energy_kcal_100g": 512, "sugars_100g": 34, "saturated_fat_100g": 12,
              "fat_100g": 30, "carbohydrates_100g": 55, "salt_100g": 1.2,
              "proteins_100g": 8, "sodium_100g": 0.48, "fiber_100g": 3}
    ings = ["wheat flour", "sugar", "soy lecithin"]
    products = [
        OFFProduct.from_json(make_off_product("111", gt_nut, ings, "e")),
        OFFProduct.from_json(make_off_product("222", gt_nut, ings, "e")),
    ]
    # Fake extractor returns exactly the GT -> high extraction scores.
    ex = FakeExtractor(nutrition=gt_nut, ingredients=ings)
    pipe = Pipeline(extractor=ex)
    report = run_eval(products, pipeline=pipe, image_loader=lambda p: b"img", profiles=["general"])

    assert report.n_products == 2
    assert report.nutrition["f1"] > 0.9          # extractor matches GT
    assert report.ingredients["f1"] == 1.0
    # every injection type should be applicable (GT has all needed fields) and caught
    assert report.validation_catch_rate["overall"]["rate"] == 1.0
    assert report.score_agreement["accuracy"] == 1.0   # base grade E == OFF grade E
    assert report.backend == "fake"


def test_run_eval_no_images_still_measures_catch_rate():
    gt_nut = {"carbohydrates_100g": 50, "proteins_100g": 10, "fat_100g": 15,
              "sodium_100g": 0.2, "energy_kcal_100g": 375}
    products = [OFFProduct.from_json(make_off_product("333", gt_nut, ["oats"], "b"))]
    report = run_eval(products, pipeline=Pipeline(extractor=FakeExtractor()),
                      image_loader=lambda p: None)
    assert report.n_products == 0
    assert report.validation_catch_rate["overall"]["applicable"] > 0
