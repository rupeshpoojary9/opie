from conftest import panel

from opie.deception.adjudicator import Adjudicator
from opie.deception.claims import normalize_claim
from opie.deception.eval import run_deception_eval
from opie.feedback.corpus import synthetic_corpus
from opie.schemas import ClaimStatus, Ingredient, Severity
from opie.taxonomy.taxonomy import normalize_ingredients

ADJ = Adjudicator()


def _ings(*names):
    return normalize_ingredients([Ingredient(name=n, rank=i + 1) for i, n in enumerate(names)])


def _status(claim, nutrition=None, ingredients=None, grade=None):
    return ADJ.adjudicate(claim, nutrition or panel(), ingredients or [], grade).status


# --- claim normalization ---------------------------------------------------

def test_normalize_claim_maps_freetext():
    assert normalize_claim("SUGAR FREE") == "sugar_free"
    assert normalize_claim("No Added Sugar") == "no_added_sugar"
    assert normalize_claim("High in Protein") == "high_protein"
    assert normalize_claim("100% Natural") == "natural"
    assert normalize_claim("gluten-free") == "gluten_free"
    assert normalize_claim("source of fibre") == "source_of_fibre"


def test_normalize_claim_unknown_is_none():
    assert normalize_claim("tastes amazing") is None
    assert normalize_claim("") is None


# --- regulatory rules ------------------------------------------------------

def test_sugar_free_rule():
    assert _status("sugar free", panel(sugars_100g=0.2)) == ClaimStatus.supported
    assert _status("sugar free", panel(sugars_100g=12)) == ClaimStatus.misleading
    assert _status("sugar free", panel()) == ClaimStatus.unverifiable  # no sugars extracted


def test_no_added_sugar_rule():
    assert _status("no added sugar", panel(sugars_100g=3), _ings("honey", "oats")) == ClaimStatus.misleading
    assert _status("no added sugar", panel(sugars_100g=3), _ings("oats", "water")) == ClaimStatus.supported
    assert _status("no added sugar", panel(sugars_100g=3), []) == ClaimStatus.unverifiable


def test_low_fat_and_fat_free():
    assert _status("low fat", panel(fat_100g=2)) == ClaimStatus.supported
    assert _status("low fat", panel(fat_100g=20)) == ClaimStatus.misleading
    assert _status("fat free", panel(fat_100g=0.3)) == ClaimStatus.supported


def test_high_protein_energy_fraction():
    assert _status("high in protein", panel(proteins_100g=10, energy_kcal_100g=100)) == ClaimStatus.supported
    assert _status("high in protein", panel(proteins_100g=2, energy_kcal_100g=400)) == ClaimStatus.misleading


def test_fibre_rules():
    assert _status("source of fibre", panel(fiber_100g=4, energy_kcal_100g=300)) == ClaimStatus.supported
    assert _status("high fibre", panel(fiber_100g=1, energy_kcal_100g=400)) == ClaimStatus.misleading


def test_low_salt_via_sodium_and_salt():
    assert _status("low salt", panel(sodium_100g=0.05)) == ClaimStatus.supported
    assert _status("low salt", panel(salt_100g=1.5)) == ClaimStatus.misleading   # sodium 0.6 > 0.12


def test_gluten_free_rule():
    assert _status("gluten free", panel(), _ings("wheat flour", "sugar")) == ClaimStatus.misleading
    assert _status("gluten free", panel(), _ings("rice", "water")) == ClaimStatus.supported


def test_vegan_rule():
    assert _status("vegan", panel(), _ings("milk", "sugar")) == ClaimStatus.misleading
    assert _status("vegan", panel(), _ings("water", "sugar")) == ClaimStatus.supported


def test_natural_rule_flags_additives():
    assert _status("100% natural", panel(), _ings("citric acid", "water")) == ClaimStatus.misleading
    assert _status("100% natural", panel(), _ings("water", "tomato")) == ClaimStatus.supported


def test_unverifiable_claims():
    assert _status("organic", panel(sugars_100g=5)) == ClaimStatus.unverifiable
    assert _status("fortified", panel(sugars_100g=5)) == ClaimStatus.unverifiable


# --- verdict explainability + health halo ----------------------------------

def test_misleading_verdict_is_explainable():
    v = ADJ.adjudicate("sugar free", panel(sugars_100g=30), [])
    assert v.status == ClaimStatus.misleading
    assert v.offending_fact and "30" in v.offending_fact
    assert "1924/2006" in v.basis          # cites the regulation
    assert v.severity == Severity.high
    assert v.reasons


def test_health_halo_on_poor_profile():
    good = panel(proteins_100g=10, energy_kcal_100g=100, sugars_100g=40)
    halo = ADJ.adjudicate("high in protein", good, [], base_grade="E")
    assert halo.status == ClaimStatus.supported
    assert halo.health_halo is True
    assert any("Health-halo" in r for r in halo.reasons)

    clean = ADJ.adjudicate("high in protein", panel(proteins_100g=10, energy_kcal_100g=100, sugars_100g=1),
                           [], base_grade="A")
    assert clean.health_halo is False


# --- eval harness ----------------------------------------------------------

def test_deception_eval_oracle_is_self_consistent():
    products = synthetic_corpus(80, seed=5)
    report = run_deception_eval(products, mode="oracle")
    assert report.misleading_recall == 1.0     # rules are self-consistent on true facts
    assert report.per_class["UNVERIFIABLE"]["precision"] == 1.0


def test_deception_eval_extracted_catches_most_deception():
    products = synthetic_corpus(200, seed=5)
    report = run_deception_eval(products, mode="extracted")
    assert report.n_cases > 0
    assert report.misleading_recall >= 0.85    # headline acceptance: catch deceptive claims
    assert report.passed() is True
    assert len(report.worked_examples) >= 2
    # confusion matrix totals equal case count
    total = sum(report.confusion[g][p] for g in report.confusion for p in report.confusion[g])
    assert total == report.n_cases
