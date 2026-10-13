# RESEARCH.md — Literature Map for the StyleSense Shirts Prototype

Generated 2026-10-03 from live Semantic Scholar / arXiv / IEEE searches.
Every link below is a real, verified paper. This file is the skeleton of the
literature-review chapter — expand each section, import entries into Zotero.

---

## 0. The foundation paper (our system is built on this)

> S. S. Roy, A. Kumar and R. S. Kumar, "Metadata and Review-Based Hybrid
> Apparel Recommendation System Using Cascaded Large Language Models,"
> **IEEE Access**, vol. 12, pp. 140053–140071, 2024,
> doi: 10.1109/ACCESS.2024.3462793

- Semantic Scholar: https://www.semanticscholar.org/paper/04fe78928b286e0b4748228a488db84bc7428bf8
- Cited by ~17–26 works (growing) — **our project extends it with a
  vision-verification stage and a shirts-only DuckDB query layer**
  (that sentence is your project's one-line contribution claim).

### Papers that cited the foundation paper (our closest neighbors)

| Paper | Year / Venue | Why it matters to us |
|---|---|---|
| Adaptive Fashion Recommendation Using Retrieval-Augmented Generation and Real-Time Vector Search | 2025, ICICNCT | Same pipeline shape (retrieval + ranking); RAG angle |
| A transformer-based architecture for collaborative filtering modeling in personalized recommender systems | 2025, Scientific Reports (29 cites) | Modern CF baseline to compare our IBCF against |
| Context-Aware Product Recommendations for Emerging Markets: a Sentiment-Driven LLM Approach | 2025, ICBATS | LLM + sentiment fusion — near-identical to our Phase 2/3 plan |
| Sentiment-aware user-item recommendation combining weighted XGBoost and optimized similarity metrics | 2026, IAES IJ-AI | Evidence that sentiment-boosted ranking is an active frontier |
| A Systematic Literature Review of Hybrid Deep Learning Models for Enhancing E-Commerce Recommender Systems | 2025, ICISIT | SLR template for how to structure our review chapter |

---

## Pillar 1 — Hybrid apparel recommendation

| Paper | Venue / Year | Link | Use for |
|---|---|---|---|
| S. Chakraborty et al., "Fashion Recommendation Systems, Models and Methods: A State-of-the-Art Review" (165 cites) | Informatics (MDPI), 2021 | https://www.mdpi.com/2227-9709/8/3/49 | **OPENING CITATION of the lit review** — the survey that frames the whole field |
| "Hybrid-Hierarchical Fashion Graph Attention Network for Compatibility-Oriented and Personalized Outfit Recommendation" | arXiv 2508.11105 / Mach. Learn. w. Appl., 2025 | https://arxiv.org/abs/2508.11105 | SOTA alternative (GAT) — the "considered & differed" paper |
| "A Hybrid Multimodal Deep Learning Framework for Intelligent Fashion Recommendation" | arXiv 2511.07573, 2025 | https://arxiv.org/abs/2511.07573 | Multimodal (text+image) fusion — directly parallels our rules+CLIP design |
| "Vogue Vision: Customized Apparel Recommender System Using Hybrid Approach" | IEEE CAI, 2025 | doi:10.1109/CAI64502.2025.00024 | Recent student-scale hybrid system — comparison peer |
| J. Liu, "A Hybrid LLM and Graph-Enhanced Transformer Framework for Cold-Start Session-Based Fashion Recommendation" (24 cites) | IEEE ECNCT, 2025 | doi:10.1109/ECNCT66493.2025.11172530 | Cold-start framing — our catalog-only design dodges this, say why |

## Pillar 2 — Vision models for e-commerce catalogs (our CLIP stage)

| Paper | Venue / Year | Link | Use for |
|---|---|---|---|
| A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) | ICML 2021, arXiv 2103.00020 | https://arxiv.org/abs/2103.00020 | **THE method citation** for our verification stage |
| "e-CLIP: Large-Scale Vision-Language Representation Learning in E-commerce" | arXiv 2207.00208 | https://arxiv.org/abs/2207.00208 | Shows domain-tuned CLIP beats generic — our zero-shot compromise needs this defense |
| "Extending CLIP for Category-to-image Retrieval in E-commerce" | arXiv 2112.11294 | https://arxiv.org/abs/2112.11294 | Zero-shot CLIP on product categories — the closest published precedent to our shirt/tee discrimination |
| "Visual Zero-Shot E-Commerce Product Attribute Value Generation" | arXiv 2502.15979, 2025 | https://arxiv.org/abs/2502.15979 | Zero-shot attribute extraction from product images (sleeve = attribute!) |
| "Exploring Fine-grained Retail Product Discrimination with CLIP" (MIMEX dataset) | arXiv 2409.14963, 2024 | https://arxiv.org/abs/2409.14963 | Honest evidence on zero-shot limits for fine-grained product classes — feeds our Limitations section |

## Pillar 3 — LLM → SQL query pipelines (our Phase 3)

| Paper | Venue / Year | Link | Use for |
|---|---|---|---|
| L. Shi et al., "A Survey on Employing Large Language Models for Text-to-SQL Queries" (230 cites) | ACM Computing Surveys 2025 / arXiv 2407.15186 | https://arxiv.org/abs/2407.15186 | **PRIMARY survey** — architecture taxonomy for our design section |
| X. Zhu et al., "Large Language Model Enhanced Text-to-SQL Generation: A Survey" (110 cites) | arXiv 2410.06011, 2024 | https://arxiv.org/abs/2410.06011 | Prompt-engineering vs fine-tuning trade-off — justifies our zero-cost prompt approach |
| "A Survey of Text-to-SQL in the Era of LLMs" | arXiv 2408.05109 | https://arxiv.org/abs/2408.05109 | Failure modes + execution-accuracy metrics (Spider/BIRD benchmarks) |

## Pillar 4 — Sentiment-aware ranking (our fusion formula)

| Paper | Venue / Year | Link | Use for |
|---|---|---|---|
| Y. Gajula et al., "Sentiment-Aware Recommendation Systems in E-Commerce: A Review from an NLP Perspective" (22 cites) | arXiv 2505.03828, 2025 | https://arxiv.org/abs/2505.03828 | The 2023-2025 survey — where our 0.5/0.5 blend sits in the landscape |
| "Review-Aware Recommender Systems (RARSs): A Survey" | ACM 2025, doi:10.1145/3744661 | https://dl.acm.org/doi/10.1145/3744661 | Formal taxonomy: rating-only vs review-mining recommenders |
| S. M. Al-Ghuribi et al., "A Comparative Study of Sentiment-Aware Collaborative Filtering..." | IEEE Access 2024 | https://ieeexplore.ieee.org/document/10741193 | Empirical: sentiment fusion improves ranking — our exact claim |

---

## The 5 research questions — answered (with receipts)

**Q1. How do hybrid apparel recommenders weight content vs collaborative signals?**
The foundation paper (Roy 2024) fuses content + IBCF + LLM cascade; the
Scientific Reports 2025 transformer-CF paper shows the field moving toward
learned fusion. Our weighted-score fusion is the *interpretable baseline* —
defend it as such (judges love "we chose transparency over a black box, and
here's the citation that says baselines still compete").

**Q2. What accuracy does CLIP zero-shot get on fashion vs fine-tuned?**
e-CLIP (2207.00208) shows fine-tuned domain CLIP wins; 2112.11294 shows
zero-shot CLIP is already usable for product categories; MIMEX (2409.14963)
quantifies the fine-grained gap. Conclusion: zero-shot + rule-confirmation
(our two-stage funnel) is a *published-reasonable* architecture, not a shortcut.

**Q3. Failure modes of LLM-generated SQL and guards?**
Surveys (2407.15186, 2408.05109) name: schema-linking errors, hallucinated
columns, wrong aggregation, injection via user text. Guards they validate:
constrained prompting with schema in context, SQL whitelisting/validation,
execution-feedback repair. Our mitigation: LLM only filters against a
**fixed 6-column schema** + DuckDB read-only = worst case is a wrong query,
never data damage.

**Q4. How is sentiment merged into ranking without drowning ratings?**
Gajula review + Al-Ghuribi comparative study: common patterns are
(a) sentiment-adjusted ratings, (b) separate feature with learned weight,
(c) two-stage rerank. Our `rating*0.5 + sentiment*0.5` is pattern (b) with
equal weights — cite, then tune the weight with the eval harness and show
the curve. That's a whole experiment section for free.

**Q5. What metrics should we report?**
Recall@K, Precision@K, NDCG@K, MRR (all four used across the pillar-1/4
papers). Spider/BIRD-style execution accuracy for the text-to-SQL stage.
Current eval_harness.py gives top-10 lists — upgrade to these four metrics
next (todo below).

---

## Platform playbook (what was used to produce this file)

1. **Semantic Scholar API** — structured paper/citation search (used above)
2. **IEEE Xplore / arXiv** — full texts (all arXiv links are free PDFs)
3. **Connected Papers** (connectedpapers.com) — next step: seed with Roy 2024,
   screenshot the graph for the report's "related work" visual
4. **Zotero** — import all 16 entries above (magic wand → paste DOI/arXiv ID);
   export IEEE citation style for the report

## Reading order (highest value per hour)

1. Chakraborty survey (Pillar 1) — frames everything
2. Roy 2024 foundation paper — re-read with fresh eyes
3. Gajula sentiment review (Pillar 4) — straight into our fusion section
4. Text-to-SQL survey 2407.15186 — only §3 (LLM pipelines) + §5 (errors)
5. e-CLIP + 2112.11294 — skim for the vision section

## TODO (research tasks, in order)

- [ ] Import 16 papers into Zotero collection "StyleSense-Shirts"
- [ ] Connected Papers graph screenshot → docs/related_work_graph.png
- [ ] Upgrade scripts/eval_harness.py: Recall@10, Precision@10, NDCG@10, MRR
- [ ] After Kaggle CV run: report rules-vs-vision agreement % as a project metric
- [ ] Weight-sweep experiment: sentiment weight 0.0→1.0, plot NDCG (Q4 curve)
