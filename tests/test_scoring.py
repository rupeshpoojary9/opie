from conftest import panel

from opie.scoring.nutriscore import nutriscore
from opie.scoring.personalization import PROFILES, personalize
from opie.schemas import TrafficLight


def test_nutriscore_unhealthy_is_e():
    p = panel(energy_kcal_100g=500, sugars_100g=45, saturated_fat_100g=15, salt_100g=2.0)
    points, grade = nutriscore(p)
    assert grade == "E"
    assert points is not None and points >= 19


def test_nutriscore_healthy_is_a():
    p = panel(energy_kcal_100g=50, sugars_100g=2, saturated_fat_100g=0.5,
              salt_100g=0.05, fiber_100g=6, proteins_100g=10)
    points, grade = nutriscore(p)
    assert grade == "A"
    assert points is not None and points <= -1


def test_nutriscore_no_data_is_none():
    points, grade = nutriscore(panel())
    assert points is None and grade is None


def test_personalize_diabetic_flags_sugar_red():
    p = panel(sugars_100g=20, carbohydrates_100g=40)
    score = personalize(p, "diabetic")
    assert score.flag == TrafficLight.red
    assert any("sugars" in r for r in score.why)


def test_personalize_general_clean_is_green():
    p = panel(sugars_100g=1, saturated_fat_100g=0.2, salt_100g=0.05)
    score = personalize(p, "general")
    assert score.flag == TrafficLight.green


def test_all_profiles_runnable():
    p = panel(sugars_100g=5, saturated_fat_100g=2, salt_100g=0.5,
              energy_kcal_100g=200, carbohydrates_100g=30, sodium_100g=0.2)
    for prof in PROFILES:
        score = personalize(p, prof)
        assert score.profile == prof
        assert score.flag in TrafficLight
