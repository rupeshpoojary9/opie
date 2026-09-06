"""Pipeline assembly.

The canonical implementation wires the nodes into a LangGraph StateGraph with the three
extraction agents fanning out from OCR and fanning back into assembly. If LangGraph isn't
importable, a pure-Python sequential runner with identical semantics is used, so the rest
of the system (API, eval, tests) never depends on the graph library being present.
"""
from __future__ import annotations

import time
from typing import Optional

from opie.config import SETTINGS, Settings
from opie.llm import build_extractor
from opie.llm.base import Extractor
from opie.pipeline.nodes import Nodes, PipelineState
from opie.rules.engine import RulesEngine
from opie.schemas import ProductIntelligence
from opie.taxonomy.taxonomy import Taxonomy


class Pipeline:
    def __init__(
        self,
        extractor: Optional[Extractor] = None,
        settings: Settings = SETTINGS,
        rules: Optional[RulesEngine] = None,
        taxonomy: Optional[Taxonomy] = None,
    ) -> None:
        self.settings = settings
        self.extractor = extractor or build_extractor(settings)
        self.nodes = Nodes(self.extractor, rules or RulesEngine(), taxonomy or Taxonomy())
        self._graph = self._try_build_graph()

    # --- public -----------------------------------------------------------

    def run(self, product_id: str, image_bytes: bytes, profiles: Optional[list[str]] = None) -> ProductIntelligence:
        try:
            self.extractor.reset_cost()
        except Exception:
            pass
        state: PipelineState = {
            "product_id": product_id,
            "image_bytes": image_bytes,
            "profiles": profiles or ["general"],
        }
        started = time.perf_counter()
        if self._graph is not None:
            out = self._graph.invoke(state)
        else:
            out = self._run_sequential(state)
        result: ProductIntelligence = out["result"]
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    @property
    def engine(self) -> str:
        return "langgraph" if self._graph is not None else "sequential"

    # --- LangGraph path ---------------------------------------------------

    def _try_build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception:
            return None
        n = self.nodes
        g = StateGraph(PipelineState)
        g.add_node("ocr", n.ocr)
        g.add_node("nutrition", n.nutrition_agent)
        g.add_node("ingredients", n.ingredient_agent)
        g.add_node("claims", n.claims_agent)
        g.add_node("taxonomy", n.taxonomy_node)
        g.add_node("assemble", n.assemble)

        g.add_edge(START, "ocr")
        # fan out to the three extraction agents
        g.add_edge("ocr", "nutrition")
        g.add_edge("ocr", "ingredients")
        g.add_edge("ocr", "claims")
        # ingredients -> taxonomy; nutrition/claims join at assemble
        g.add_edge("ingredients", "taxonomy")
        g.add_edge("nutrition", "assemble")
        g.add_edge("claims", "assemble")
        g.add_edge("taxonomy", "assemble")
        g.add_edge("assemble", END)
        return g.compile()

    # --- sequential fallback (identical semantics) ------------------------

    def _run_sequential(self, state: PipelineState) -> dict:
        n = self.nodes
        state.update(n.ocr(state))
        state.update(n.nutrition_agent(state))
        state.update(n.ingredient_agent(state))
        state.update(n.claims_agent(state))
        state.update(n.taxonomy_node(state))
        state.update(n.assemble(state))
        return state
