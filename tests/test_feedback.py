from conftest import panel

from opie.feedback.calibration import CalibratorSet
from opie.feedback.corpus import synthetic_corpus
from opie.feedback.loop import FeedbackLoop
from opie.feedback.memory import CorrectionMemory
from opie.feedback.oracle import ReviewOracle
from opie.feedback.report import FeedbackReport
from opie.feedback.simulate import SimulatedExtractor
from opie.feedback.store import FeedbackStore
from opie.feedback.types import CorrectionCase, FieldObservation
from opie.feedback.vectorize import cosine, embed


# --- retrieval / vectorize -------------------------------------------------

def test_embed_deterministic_and_cosine():
    f = {"name": "sugars_100g", "category": "soda", "mag": "0", "unit": "g"}
    v1, v2 = embed(f), embed(f)
    assert v1 == v2
    assert abs(cosine(v1, v2) - 1.0) < 1e-9


# --- correction memory (learn from corrections) ----------------------------

def test_memory_learns_numeric_scale_transform():
    mem = CorrectionMemory(min_support=3)
    # five soda/sugars cases where the extractor under-read by 10x (correct = 10 * extracted)
    for i in range(5):
        mem.add(CorrectionCase(
            product_id=f"p{i}", kind="nutrition", name="sugars_100g",
            extracted_value=1.0 + i * 0.1, correct_value=10.0 + i,
            features={"name": "sugars_100g", "category": "soda", "mag": "0", "unit": "g"}))
    corrected, support = mem.correct_numeric(
        1.2, {"name": "sugars_100g", "category": "soda", "mag": "0", "unit": "g"})
    assert corrected is not None
    assert support >= 3
    assert 10.0 < corrected < 14.0   # ~ *10


def test_memory_learns_token_map_and_spurious():
    mem = CorrectionMemory()
    for i in range(2):
        mem.add(CorrectionCase(product_id=f"p{i}", kind="ingredient", name="ing1",
                               extracted_value="sugarr", correct_value="sugar",
                               features={"name": "ingredient", "category": "biscuits"}))
        mem.add(CorrectionCase(product_id=f"q{i}", kind="ingredient", name="ing2",
                               extracted_value="e000 unknown", correct_value=None,
                               features={"name": "ingredient", "category": "biscuits"}))
    assert mem.correct_token("sugarr") == ("map", "sugar")
    assert mem.correct_token("e000 unknown") == ("drop", None)
    assert mem.correct_token("water") == ("keep", "water")


# --- calibration -----------------------------------------------------------

def test_calibrator_fits_and_lowers_confidence_for_bad_band():
    obs = []
    # low-confidence band is mostly wrong; high-confidence band mostly right
    for i in range(40):
        obs.append(FieldObservation("p", "nutrition", "sugars_100g", 1.0, 0.4,
                                    is_correct=(i % 5 == 0)))          # 20% correct
    for i in range(40):
        obs.append(FieldObservation("p", "nutrition", "sugars_100g", 1.0, 0.9,
                                    is_correct=(i % 10 != 0)))         # 90% correct
    cal = CalibratorSet()
    cal.fit(obs)
    assert cal.calibrate("sugars_100g", 0.4) < cal.calibrate("sugars_100g", 0.9)
    assert cal.calibrate("sugars_100g", 0.4) < 0.5


# --- oracle ----------------------------------------------------------------

def test_oracle_labels_nutrition_and_ingredients():
    prod = synthetic_corpus(8, seed=1)[0]
    oracle = ReviewOracle()
    field = next(iter(prod.gt_nutrition))
    good = FieldObservation(prod.code, "nutrition", field, prod.gt_nutrition[field], 0.9)
    oracle.label(good, prod)
    assert good.is_correct is True

    bad = FieldObservation(prod.code, "nutrition", field, prod.gt_nutrition[field] * 5, 0.5)
    oracle.label(bad, prod)
    assert bad.is_correct is False
    assert oracle.correct_value(bad, prod) == prod.gt_nutrition[field]


# --- simulated extractor ---------------------------------------------------

def test_simulated_extractor_deterministic():
    prod = synthetic_corpus(8, seed=2)[0]
    ex = SimulatedExtractor()
    a = ex.extract(prod)
    b = ex.extract(prod)
    assert [(o.name, o.extracted_value, o.raw_confidence) for o in a] == \
           [(o.name, o.extracted_value, o.raw_confidence) for o in b]


# --- store -----------------------------------------------------------------

def test_store_roundtrip(tmp_path):
    store = FeedbackStore(db_path=tmp_path / "fb.sqlite")
    store.start_run("run1", 0, "learned", 1)
    obs = [FieldObservation("p", "nutrition", "sugars_100g", 34.0, 0.5, queued=True, is_correct=False)]
    store.save_observations("run1", obs)
    store.save_corrections([CorrectionCase("p", "nutrition", "sugars_100g", 3.4, 34.0)])
    counts = store.counts()
    assert counts["runs"] == 1 and counts["observations"] == 1 and counts["corrections"] == 1
    assert len(store.review_queue("run1")) == 1


# --- the whole flywheel: F1 rises, queue shrinks, beats baseline -----------

def test_feedback_loop_lifts_accuracy_and_shrinks_queue():
    products = synthetic_corpus(300, seed=7)
    loop = FeedbackLoop(products, rounds=5, seed=13)
    rounds = loop.run()
    report = FeedbackReport(rounds)

    assert report.last.f1 > report.first.f1            # rises across rounds
    assert report.last.f1 > report.last.base_f1        # beats frozen baseline
    assert report.last.queue_size < report.first.queue_size   # queue shrinks
    assert report.last.ece <= report.first.ece         # calibration no worse
    assert report.passed() is True
    # memory actually accumulated corrections
    assert report.last.memory_size > 0


def test_feedback_rounds_are_disjoint():
    products = synthetic_corpus(50, seed=3)
    loop = FeedbackLoop(products, rounds=5, seed=13)
    seen = set()
    for bucket in loop._round_products:
        codes = {p.code for p in bucket}
        assert not (codes & seen)     # no product appears in two rounds
        seen |= codes
    assert len(seen) == 50


def test_baseline_stays_flat():
    products = synthetic_corpus(300, seed=7)
    rounds = FeedbackLoop(products, rounds=5, seed=13).run()
    base = [r.base_f1 for r in rounds]
    assert max(base) - min(base) < 0.06     # frozen baseline barely moves
