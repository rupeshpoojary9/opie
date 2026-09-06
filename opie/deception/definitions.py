"""Regulatory definitions for nutrition/content claims — PUBLIC standards, cited in code.

Thresholds are taken from public regulation and standards, not any proprietary source:
  * EU Reg (EC) No 1924/2006 on nutrition and health claims, Annex (conditions per claim)
  * Codex Alimentarius CAC/GL 23-1997, Guidelines for Use of Nutrition & Health Claims
  * FSSAI Food Safety and Standards (Labelling and Display) Regulations, 2020, Schedule
  * EU Reg (EU) No 1169/2011 Art. 7 (fair information; basis for "natural"/"vegan")
  * Codex Standard 118-1979 / Reg (EU) 828/2014 ("gluten-free")

Each definition returns an explainable RuleOutcome. Values are per 100 g (solids); the
per-100 ml liquid variant is noted in the citation where it differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from opie.deception import claims as C
from opie.schemas import ClaimStatus, Ingredient, NutritionPanel, Severity

# Public regulatory citation shared by the nutrient-content claims.
_EU_CODEX_FSSAI = ("EU Reg (EC) 1924/2006 Annex; Codex CAC/GL 23-1997; "
                   "FSSAI Labelling & Display Regs 2020")

# Ingredients that constitute *added sugar* (public/common knowledge).
ADDED_SUGAR_INGREDIENTS = (
    "sugar", "glucose", "fructose", "glucose-fructose syrup", "glucose syrup",
    "corn syrup", "high fructose corn syrup", "honey", "dextrose", "maltose",
    "molasses", "invert sugar", "fruit juice concentrate", "malt extract", "malt",
    "agave", "cane juice", "cane sugar", "syrup", "sucrose", "maltodextrin",
)
GLUTEN_GRAINS = ("wheat", "barley", "rye", "oat", "spelt", "malt", "semolina", "farro", "kamut")
ANIMAL_INGREDIENTS = (
    "milk", "whey", "casein", "cheese", "butter", "cream", "lactose", "egg", "albumen",
    "honey", "gelatin", "gelatine", "fish", "anchovy", "meat", "chicken", "beef", "pork",
    "lard", "carmine", "shellac", "lard",
)


@dataclass
class Facts:
    """The extracted facts a definition reads."""
    nutrition: NutritionPanel
    ingredients: list[Ingredient] = field(default_factory=list)

    def value(self, field_name: str) -> Optional[float]:
        return getattr(self.nutrition, field_name).value

    def evidence(self, field_name: str) -> Optional[str]:
        ev = getattr(self.nutrition, field_name).evidence
        return ev.text if ev else None

    @property
    def energy_kcal(self) -> Optional[float]:
        return self.value("energy_kcal_100g")

    def sodium_g(self) -> Optional[float]:
        s = self.value("sodium_100g")
        if s is not None:
            return s
        salt = self.value("salt_100g")
        return salt / 2.5 if salt is not None else None

    def ingredient_names(self) -> list[str]:
        return [i.name.lower() for i in self.ingredients]

    def _match(self, needles) -> Optional[str]:
        for name in self.ingredient_names():
            for n in needles:
                if n in name:
                    return name
        return None

    def added_sugar_ingredient(self) -> Optional[str]:
        return self._match(ADDED_SUGAR_INGREDIENTS)

    def gluten_ingredient(self) -> Optional[str]:
        return self._match(GLUTEN_GRAINS)

    def animal_ingredient(self) -> Optional[str]:
        return self._match(ANIMAL_INGREDIENTS)

    def additive_or_ultraprocessed(self) -> Optional[str]:
        for i in self.ingredients:
            if i.enumber or i.ultra_processing_marker:
                return i.name.lower()
        return None


@dataclass
class RuleOutcome:
    status: ClaimStatus
    basis: str
    offending_fact: Optional[str] = None
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    severity: Severity = Severity.low


@dataclass
class Definition:
    claim_id: str
    citation: str
    description: str
    check: Callable[[Facts], RuleOutcome]


# --- helpers ---------------------------------------------------------------

def _unverifiable(citation: str, reason: str) -> RuleOutcome:
    return RuleOutcome(ClaimStatus.unverifiable, citation, reasons=[reason])


def _max_threshold(cid: str, field_name: str, limit: float, label: str,
                   unit: str = "g", severity: Severity = Severity.high) -> Callable[[Facts], RuleOutcome]:
    """Claim holds iff facts[field] <= limit."""
    def check(f: Facts) -> RuleOutcome:
        v = f.value(field_name)
        if v is None:
            return _unverifiable(_EU_CODEX_FSSAI, f"{field_name} not extracted; cannot verify {cid}.")
        ev = [e for e in [f.evidence(field_name)] if e]
        if v <= limit:
            return RuleOutcome(ClaimStatus.supported, _EU_CODEX_FSSAI,
                               reasons=[f"{label} {v:g}{unit} ≤ {limit:g}{unit} limit."], evidence=ev)
        return RuleOutcome(ClaimStatus.misleading, _EU_CODEX_FSSAI,
                           offending_fact=f"{label} {v:g}{unit} exceeds the {limit:g}{unit} '{cid}' limit.",
                           reasons=[f"'{cid}' requires {label} ≤ {limit:g}{unit}; extracted {v:g}{unit}."],
                           evidence=ev, severity=severity)
    return check


def _protein_energy_fraction(cid: str, min_frac: float) -> Callable[[Facts], RuleOutcome]:
    def check(f: Facts) -> RuleOutcome:
        prot, energy = f.value("proteins_100g"), f.energy_kcal
        if prot is None or energy is None or energy <= 0:
            return _unverifiable(_EU_CODEX_FSSAI, f"protein and/or energy not extracted; cannot verify {cid}.")
        frac = (prot * 4.0) / energy
        ev = [e for e in [f.evidence("proteins_100g"), f.evidence("energy_kcal_100g")] if e]
        pct = frac * 100
        if frac >= min_frac:
            return RuleOutcome(ClaimStatus.supported, _EU_CODEX_FSSAI,
                               reasons=[f"protein supplies {pct:.0f}% of energy ≥ {min_frac * 100:.0f}%."], evidence=ev)
        return RuleOutcome(ClaimStatus.misleading, _EU_CODEX_FSSAI,
                           offending_fact=f"protein supplies only {pct:.0f}% of energy (need ≥ {min_frac * 100:.0f}%).",
                           reasons=[f"'{cid}' requires ≥ {min_frac * 100:.0f}% of energy from protein; got {pct:.0f}%."],
                           evidence=ev, severity=Severity.high)
    return check


def _fibre(cid: str, per100g: float, per100kcal: float) -> Callable[[Facts], RuleOutcome]:
    def check(f: Facts) -> RuleOutcome:
        fib, energy = f.value("fiber_100g"), f.energy_kcal
        if fib is None:
            return _unverifiable(_EU_CODEX_FSSAI, f"fibre not extracted; cannot verify {cid}.")
        ev = [e for e in [f.evidence("fiber_100g")] if e]
        ok = fib >= per100g or (energy and energy > 0 and (fib / energy) * 100 >= per100kcal)
        if ok:
            return RuleOutcome(ClaimStatus.supported, _EU_CODEX_FSSAI,
                               reasons=[f"fibre {fib:g}g/100g meets the '{cid}' threshold "
                                        f"(≥{per100g:g}g/100g or ≥{per100kcal:g}g/100kcal)."], evidence=ev)
        return RuleOutcome(ClaimStatus.misleading, _EU_CODEX_FSSAI,
                           offending_fact=f"fibre {fib:g}g/100g below the '{cid}' threshold ({per100g:g}g/100g).",
                           reasons=[f"'{cid}' requires ≥{per100g:g}g fibre/100g (or ≥{per100kcal:g}g/100kcal)."],
                           evidence=ev, severity=Severity.medium)
    return check


def _ingredient_absent(cid: str, finder: str, citation: str, kind: str,
                       severity: Severity) -> Callable[[Facts], RuleOutcome]:
    """Claim holds iff no ingredient of the given kind is present."""
    def check(f: Facts) -> RuleOutcome:
        if not f.ingredients:
            return _unverifiable(citation, f"ingredient list not extracted; cannot verify {cid}.")
        hit = getattr(f, finder)()
        if hit is None:
            return RuleOutcome(ClaimStatus.supported, citation,
                               reasons=[f"no {kind} ingredient found in the list."])
        return RuleOutcome(ClaimStatus.misleading, citation,
                           offending_fact=f"ingredient '{hit}' is a {kind} source.",
                           reasons=[f"'{cid}' is contradicted by ingredient '{hit}'."],
                           evidence=[hit], severity=severity)
    return check


def _no_added_sugar(f: Facts) -> RuleOutcome:
    if not f.ingredients:
        return _unverifiable(_EU_CODEX_FSSAI, "ingredient list not extracted; cannot verify no_added_sugar.")
    hit = f.added_sugar_ingredient()
    if hit is None:
        return RuleOutcome(ClaimStatus.supported, _EU_CODEX_FSSAI,
                           reasons=["no added-sugar ingredient found."])
    return RuleOutcome(ClaimStatus.misleading, _EU_CODEX_FSSAI,
                       offending_fact=f"ingredient '{hit}' adds sugar.",
                       reasons=["'no added sugar' requires no added mono/disaccharides or "
                                f"sweetening ingredient; found '{hit}'."],
                       evidence=[hit], severity=Severity.high)


def _natural(f: Facts) -> RuleOutcome:
    cite = "EU Reg (EU) 1169/2011 Art. 7 (fair information); EU guidance excludes additives from 'natural'"
    if not f.ingredients:
        return _unverifiable(cite, "ingredient list not extracted; cannot verify 'natural'.")
    hit = f.additive_or_ultraprocessed()
    if hit is None:
        return RuleOutcome(ClaimStatus.supported, cite, reasons=["no additives / ultra-processing markers found."])
    return RuleOutcome(ClaimStatus.misleading, cite,
                       offending_fact=f"ingredient '{hit}' is an additive / ultra-processing marker.",
                       reasons=[f"'natural' is undermined by additive/ultra-processed ingredient '{hit}'."],
                       evidence=[hit], severity=Severity.medium)


# --- registry --------------------------------------------------------------

DEFINITIONS: dict[str, Definition] = {
    C.SUGAR_FREE: Definition(C.SUGAR_FREE, _EU_CODEX_FSSAI,
        "sugars ≤ 0.5 g/100g", _max_threshold(C.SUGAR_FREE, "sugars_100g", 0.5, "sugars")),
    C.LOW_SUGAR: Definition(C.LOW_SUGAR, _EU_CODEX_FSSAI,
        "sugars ≤ 5 g/100g (2.5 g/100ml liquids)", _max_threshold(C.LOW_SUGAR, "sugars_100g", 5.0, "sugars")),
    C.NO_ADDED_SUGAR: Definition(C.NO_ADDED_SUGAR, _EU_CODEX_FSSAI,
        "no added mono/disaccharides or sweetening ingredient", _no_added_sugar),
    C.FAT_FREE: Definition(C.FAT_FREE, _EU_CODEX_FSSAI,
        "fat ≤ 0.5 g/100g", _max_threshold(C.FAT_FREE, "fat_100g", 0.5, "fat")),
    C.LOW_FAT: Definition(C.LOW_FAT, _EU_CODEX_FSSAI,
        "fat ≤ 3 g/100g (1.5 g/100ml liquids)", _max_threshold(C.LOW_FAT, "fat_100g", 3.0, "fat")),
    C.LOW_SATURATED_FAT: Definition(C.LOW_SATURATED_FAT, _EU_CODEX_FSSAI,
        "saturated fat ≤ 1.5 g/100g", _max_threshold(C.LOW_SATURATED_FAT, "saturated_fat_100g", 1.5, "saturated fat")),
    C.SALT_FREE: Definition(C.SALT_FREE, _EU_CODEX_FSSAI,
        "sodium ≤ 0.005 g/100g",
        lambda f: _sodium_threshold(C.SALT_FREE, f, 0.005)),
    C.LOW_SALT: Definition(C.LOW_SALT, _EU_CODEX_FSSAI,
        "sodium ≤ 0.12 g/100g (salt ≤ 0.3 g/100g)",
        lambda f: _sodium_threshold(C.LOW_SALT, f, 0.12)),
    C.LOW_ENERGY: Definition(C.LOW_ENERGY, _EU_CODEX_FSSAI,
        "energy ≤ 40 kcal/100g (20 kcal/100ml liquids)",
        _max_threshold(C.LOW_ENERGY, "energy_kcal_100g", 40.0, "energy", unit=" kcal", severity=Severity.medium)),
    C.ENERGY_FREE: Definition(C.ENERGY_FREE, _EU_CODEX_FSSAI,
        "energy ≤ 4 kcal/100ml",
        _max_threshold(C.ENERGY_FREE, "energy_kcal_100g", 4.0, "energy", unit=" kcal", severity=Severity.medium)),
    C.HIGH_PROTEIN: Definition(C.HIGH_PROTEIN, _EU_CODEX_FSSAI,
        "≥ 20% of energy from protein", _protein_energy_fraction(C.HIGH_PROTEIN, 0.20)),
    C.SOURCE_OF_PROTEIN: Definition(C.SOURCE_OF_PROTEIN, _EU_CODEX_FSSAI,
        "≥ 12% of energy from protein", _protein_energy_fraction(C.SOURCE_OF_PROTEIN, 0.12)),
    C.HIGH_FIBRE: Definition(C.HIGH_FIBRE, _EU_CODEX_FSSAI,
        "≥ 6 g fibre/100g or 3 g/100kcal", _fibre(C.HIGH_FIBRE, 6.0, 3.0)),
    C.SOURCE_OF_FIBRE: Definition(C.SOURCE_OF_FIBRE, _EU_CODEX_FSSAI,
        "≥ 3 g fibre/100g or 1.5 g/100kcal", _fibre(C.SOURCE_OF_FIBRE, 3.0, 1.5)),
    C.GLUTEN_FREE: Definition(C.GLUTEN_FREE,
        "Codex Standard 118-1979; EU Reg (EU) 828/2014",
        "no gluten-containing grain ingredient (legal limit < 20 mg/kg)",
        _ingredient_absent(C.GLUTEN_FREE, "gluten_ingredient",
                           "Codex Standard 118-1979; EU Reg (EU) 828/2014", "gluten-grain", Severity.high)),
    C.VEGAN: Definition(C.VEGAN, "EU Reg (EU) 1169/2011 Art. 7 (fair information)",
        "no animal-derived ingredient",
        _ingredient_absent(C.VEGAN, "animal_ingredient",
                           "EU Reg (EU) 1169/2011 Art. 7", "animal-derived", Severity.high)),
    C.NATURAL: Definition(C.NATURAL, "EU Reg (EU) 1169/2011 Art. 7",
        "no additives / ultra-processing markers", _natural),
    C.ORGANIC: Definition(C.ORGANIC, "EU Reg (EU) 2018/848; requires accredited certification",
        "requires organic certification — not derivable from nutrition facts",
        lambda f: _unverifiable("EU Reg (EU) 2018/848",
                                "organic status requires certification, not derivable from extracted facts.")),
    C.FORTIFIED: Definition(C.FORTIFIED, "EU Reg (EC) 1925/2006 (addition of vitamins/minerals)",
        "requires added-micronutrient data — not extracted",
        lambda f: _unverifiable("EU Reg (EC) 1925/2006",
                                "fortification requires micronutrient content, which is not extracted.")),
    C.SOURCE_OF_VITAMINS: Definition(C.SOURCE_OF_VITAMINS, _EU_CODEX_FSSAI,
        "≥ 15% of NRV per 100g — requires micronutrient data",
        lambda f: _unverifiable(_EU_CODEX_FSSAI,
                                "vitamin/mineral content vs NRV is not extracted; cannot verify.")),
}


def _sodium_threshold(cid: str, f: Facts, limit: float) -> RuleOutcome:
    v = f.sodium_g()
    if v is None:
        return _unverifiable(_EU_CODEX_FSSAI, f"salt/sodium not extracted; cannot verify {cid}.")
    ev = [e for e in [f.evidence("salt_100g"), f.evidence("sodium_100g")] if e]
    if v <= limit:
        return RuleOutcome(ClaimStatus.supported, _EU_CODEX_FSSAI,
                           reasons=[f"sodium {v:.3g}g/100g ≤ {limit:g}g/100g limit."], evidence=ev)
    return RuleOutcome(ClaimStatus.misleading, _EU_CODEX_FSSAI,
                       offending_fact=f"sodium {v:.3g}g/100g exceeds the {limit:g}g/100g '{cid}' limit.",
                       reasons=[f"'{cid}' requires sodium ≤ {limit:g}g/100g; extracted {v:.3g}g/100g."],
                       evidence=ev, severity=Severity.high)
