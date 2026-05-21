# Literature Review: LLM-Assisted Free-Comment Sensory Modeling

## Scope

This review covers the literature across seven strands: Free-Comment and Ideal-Free-Comment methods, multiple-response correspondence analysis, text mining and NLP in sensory science, LLMs as judges and annotators, Semantic Similarity Rating, reasoning models and overthinking, and tabular foundation models. It reports what each strand establishes.

A structured literature map with citation roles and DOIs follows the prose synthesis, and references appear at the end.

## Synthesis

Free-Comment (FC) methods allow consumers to describe products in their own words without reliance on pre-defined attribute lists. This approach addresses limitations of structured methods such as Check-All-That-Apply (CATA), where the vocabulary is imposed in advance and may not fully capture consumer perceptions or language. Mahieu and colleagues have documented the performance and extensions of FC. In a home-use test with wines, FC outperformed CATA on discrimination power and stability of product characterization when both were applied with consumers at home (Mahieu et al., 2020). Subsequent work developed a multiple-response chi-square framework and multiple-response correspondence analysis (MR-CA) tailored to the structure of FC data, in which experimental units are individual evaluations rather than aggregated citation counts (Mahieu et al., 2021). This framework includes dimensionality testing for dependence and hypergeometric tests for descriptor-product associations.

Ideal-Free-Comment (IFC) extends FC by asking consumers to describe both the actual product and their ideal product using free text. In a home-use test involving 483 French consumers and 30 commercial cooked hams, Mahieu, Visalli and Schlich (2022) linked FC and IFC data to overall liking scores collected on a visual analogue scale. They applied mixed linear models of the form liking ~ product + descriptor presence + consumer random effect and identified sensory drivers of liking. Flavor descriptors showed the strongest associations, followed by texture and then visual appearance. The panel-level ideal product description aligned with these drivers, and consumer segmentation on ideal profiles revealed flavor-based differences. The authors report that drivers of liking are less exposed to cognitive and attitudinal bias than ideal-product descriptions, and describe a smaller, noisier ideal-product segment. Worch and colleagues developed the related Ideal Profile Method (IPM), in which consumers rate both perceived and ideal intensities on structured attribute lists; IFC can be viewed as an unstructured, free-text analogue that avoids attribute pre-selection (Worch, 2013). The cooked-ham dataset is described in detail in an open data article (Visalli et al., 2024), which reports the home-use field design and confirms the 2,758 evaluations from 483 consumers across 30 hams.

Multiple-response correspondence analysis provides a geometric representation of product-descriptor associations while respecting the multiple-response nature of FC data. Mahieu et al. (2021) positioned MR-CA within a broader multiple-response chi-square framework that includes appropriate tests for dependence and association, drawing on the multiple-response chi-square test of Loughin and Scherer (1998) and subject-stratified bootstrap procedures (Cadoret and Husson, 2013). In the cooked-ham application, MR-CA was performed separately by sensory modality (visual, texture, flavor), with the ideal product and mean liking projected as supplementary elements. The method yields maps that show which descriptors most strongly differentiate products and relate to hedonic response. MR-CA operates on discrete citation counts under a multiple-response chi-square model.

Text mining and natural language processing have been applied to sensory free-text data for lexicon development, term grouping, and automated descriptor extraction. Traditional pipelines involve cleaning, lemmatization, and frequency- or co-occurrence-based grouping, followed by statistical mapping (Hamilton and Lahne, 2020; Miller et al., 2021). A recent review summarizes sources of text data, common preprocessing choices, and the range of research questions addressed in sensory text analysis, from lexicon generation to temporal or emotional characterization (Hamilton et al., 2025). More targeted work has begun to examine whether NLP or pretrained LLMs can substitute for human operators in preprocessing word- and sentence-based free-comment data, and to build expert-system preprocessing that combines sensory lexicons with statistical methods (Visalli and Mahieu, 2023). A benchmark dataset of annotated free comments on madeleines was constructed specifically to study the impact of data-collection mode and preprocessing on FC data quality, with 200 consumers split between a group that gave words or short expressions and a group that gave full sentences (Visalli, Symoneaux et al., 2024). The associated analysis reports that preprocessing is a significant source of non-reproducibility, that human experts performed better on average than automated systems, and that automated workflows were more effective when input was given as words or short expressions rather than full sentences.

Large language models are increasingly used as automated judges for semantic evaluation tasks. LLM-as-a-judge systems prompt models to score, classify, or compare outputs according to explicit criteria, offering scalability where human annotation is costly (Zheng et al., 2023). Documented reliability concerns include positional bias, verbosity bias, and self-enhancement bias, in which a judge favors outputs resembling its own style. LLM-based evaluation can correlate with human judgment in some text-generation settings (Liu et al., 2023, G-Eval), but surveys document that reliability varies strongly by task and setup and note persistent challenges of bias, calibration, and agreement with human raters; LLM judges are not treated as general substitutes for human evaluation (Gu et al., 2024/2026). Empirically, LLM annotation has matched or exceeded crowd workers on several text-annotation tasks at substantially lower cost (Gilardi, Alizadeh and Kubli, 2023). In sensory and consumer research specifically, generative AI has been examined across the concept, design, and testing phases, with explicit attention to biased responses, privacy, oversimplification, and transparency (Motoki, Low and Velasco, 2025). LLMs have been applied to annotate coffee flavor notes (Jakkaew, Chondamrongkul and Suebsombut, 2026) and to assess creativity in flavor pairings, where GPT-4 ratings correlated strongly with human creativity ratings (r about 0.89) but agreement was weakest for rare or culturally specific flavor combinations (Wang and Pellegrino, 2025). Exploratory work has also prompted ChatGPT to act as a sensory evaluator, while noting that descriptors require validation against a human panel (Torrico, 2025). A recurring observation across these applications is that LLM reliability tends to be higher for concrete physical attributes than for abstract or hedonic concepts. Direct prompting of LLMs for numerical ratings is reported to produce narrow or central-tendency distributions that can fail to match human response variance.

Semantic Similarity Rating (SSR) addresses some of these limitations by eliciting free-text responses from LLMs and then mapping those texts to Likert-scale probability distributions via cosine similarity of embeddings to a set of reference anchor statements (Maier et al., 2025). Tested on personal care product purchase-intent surveys covering about 9,300 participants across 57 surveys, SSR recovered realistic response distributions and reached about 90 percent of human test-retest reliability without model fine-tuning. SSR preserves the qualitative richness of textual responses while producing structured outputs suitable for downstream modeling. SSR is distinct from direct LLM assignment of a single Likert score: SSR produces an embedding-derived probability mass function over the scale and relies on a fixed set of reference anchors, whereas direct scoring yields a point estimate without that embedding-projection step. A reference implementation is available from PyMC Labs.

Reasoning-enhanced LLM configurations, including explicit chain-of-thought prompting and increased test-time compute budgets, improve performance on many complex reasoning and multi-step tasks (Wei et al., 2022). A growing body of evidence indicates that additional reasoning steps or longer internal deliberation can degrade accuracy on other tasks. Inverse scaling with test-time compute has been demonstrated on synthetic tasks designed to probe specific failure modes, including distraction by irrelevant content, following of spurious correlations, and loss of focus on deductive constraints (Gema et al., 2025). Related work analyzes how the marginal utility of reasoning tokens declines and reverses as the compute budget grows, with "flip events" in which extended reasoning leads a model to abandon a previously correct answer (Zhou et al., 2026), and a survey catalogues efficient-reasoning and overthinking phenomena (Sui et al., 2025). These patterns are task-dependent: reasoning tends to help when a problem requires explicit decomposition or search, and can introduce noise on pattern-matching or perceptual judgment tasks where fast, direct assessment aligns better with human performance (Liu et al., 2024). Reasoning controls differ across model families: for example, Gemini 2.5 exposes a thinking budget while Gemini 3 exposes a thinking level, and a nominal "minimal" setting is not always equivalent to no reasoning.

Dual-process accounts of cognition distinguish fast, intuitive processing from slower, deliberative processing (Kahneman, 2011). Such frameworks are frequently invoked as analogies in prompting research, but they remain metaphorical when applied to transformer-based models and are not literal mechanistic claims about model internals.

Tabular foundation models offer an alternative to traditional machine learning pipelines for the small- to medium-sized tabular datasets common in sensory and consumer research. TabPFN (Tabular Prior-data Fitted Network) is a transformer architecture pre-trained on more than 100 million synthetic tabular datasets drawn from structural causal model priors; it performs inference via in-context learning in a single forward pass and outperforms tuned ensembles on datasets with up to roughly 10,000 samples while using substantially less computation (Hollmann et al., 2025). A later release, TabPFN-2.5, was published in November 2025 (Grinsztajn, Flöge et al., 2025), and TabICL extends the tabular-foundation-model approach to larger sample sizes (Qu et al., 2025). Because TabPFN approximates a learned inference procedure rather than fitting parameters to a single dataset, it is well suited to grouped cross-validation and to domains with limited labeled data. For comparison, gradient-boosted tree ensembles such as XGBoost and CatBoost, and AutoML stacks such as AutoGluon, remain strong baselines in the small-to-medium tabular regime.

LLM deployment involves accuracy, latency, cost, and output reliability. Reasoning configurations increase token consumption and therefore both monetary cost and wall-clock time, and can increase the rate of structured-output failures such as JSON parsing errors. Lower-temperature, direct configurations are reported to yield faster and more consistent structured responses. On model versioning, provider documentation states that "-latest" aliases are hot-swapped over time, and stable model families carry different release dates and different reasoning-control surfaces.

Two adjacent literatures are relevant as context. Product-optimization work around CATA, JAR, and ideal profiles addresses ideal characterization and reformulation guidance; relative to these list-based methods, FC elicits consumer-authored descriptors. The synthetic-respondent literature, including "silicon samples" (Argyle et al., 2023), simulated economic agents (Horton, 2023), and GPT-based willingness-to-pay (Brand, Israeli and Ngwe, 2023), addresses LLM simulation of survey respondents, including purchase-intent applications related to SSR.

## Literature map (citation roles and DOIs)

### Free-Comment, Ideal-Free-Comment, and ideal-product methods

| Reference | Relevance | Role |
|---|---|---|
| ten Kleij and Musters (2003), *Food Quality and Preference* 14(1): 43–52, doi:10.1016/S0950-3293(02)00011-3 | Foundational free-comment paper introducing text analysis of open-ended responses as a complement to preference mapping. | Background |
| Symoneaux, Galmarini and Mehinagic (2012), *Food Quality and Preference* 24(1): 59–66 | Comment analysis of likes/dislikes as a sensory tool, with chi-square per cell on contingency tables. | Background |
| Mahieu, Visalli, Thomas and Schlich (2020), *Food Quality and Preference* 84: 103937 | FC outperformed CATA in a home-use wine test; source of the cleaning, 5%-threshold descriptor retention, and contingency-table approach. | Method |
| Mahieu, Visalli and Schlich (2022), *Food Quality and Preference* 96: 104389, doi:10.1016/j.foodqual.2021.104389 | Anchor paper (three authors). Introduces Ideal-Free-Comment on the cooked-ham study (483 consumers, 30 hams, mean 5.71 hams each); mixed model identifies drivers of liking; ideal-product segment noisier. | Method, anchor |
| Visalli, Loiseau, Cordelle, Mahieu and Schlich (2024), *Data in Brief* 54: 110549, doi:10.1016/j.dib.2024.110549 | Open dataset article for the cooked-ham home-use test; documents the field design. | Data source |
| Worch, Lê, Punter and Pagès (2013), "Ideal Profile Method (IPM): the ins and outs," *Food Quality and Preference* 28(1): 45–59, doi:10.1016/j.foodqual.2012.08.001 | IPM, the structured-attribute ideal method; IFC is its unstructured free-text analogue. | Background |

### MR-CA and statistics on free-comment tables

| Reference | Relevance | Role |
|---|---|---|
| Mahieu, Schlich, Visalli and Cardot (2021), *Food Quality and Preference* 93: 104256, doi:10.1016/j.foodqual.2021.104256 | Introduces the multiple-response chi-square framework and MR-CA; underlies the cooked-ham analysis. R package MultiResponseR implements `mrCA`, `sensory.mrCA`, `mr.chisq.test`. | Method |
| Loughin and Scherer (1998), *Biometrics* 54(2): 630–637; Cadoret and Husson (2013), *FQP* 28(1): 106–115 | Multiple-response chi-square test and subject-stratified bootstrap underlying MR-CA inference. | Method |

### Text mining and NLP in sensory science

| Reference | Relevance | Role |
|---|---|---|
| Hamilton and Lahne (2020), *Food Quality and Preference* 83: 103926, doi:10.1016/j.foodqual.2020.103926 | Early sensory-NLP pipeline for automated descriptor lexicon development. | Background |
| Miller, Hamilton and Lahne (2021), *Foods* 10(7): 1633, doi:10.3390/foods10071633 | Deep-learning descriptor extraction from review corpora; high precision/recall. | Background |
| Visalli and Mahieu (2023), "Combining statistics and semantic for an automated data analysis of Free-Comment sensory description of products," 15th Pangborn Sensory Science Symposium, HAL: hal-04197590 | Expert-system preprocessing of FC text combining sensory lexicons with statistical methods. | Background |
| Visalli, Symoneaux, Mursic et al. (2025), *Data in Brief* 58: 111250, doi:10.1016/j.dib.2024.111250 | Annotated madeleines FC benchmark; words vs sentences design; basis for the preprocessing-reproducibility finding. Confirm whether the human-vs-automated and words-vs-sentences result is reported here or in a separate analysis paper. | Method, limitation |
| Hamilton, Miller and Lahne (2026), *Current Opinion in Food Science* 67: 101370, doi:10.1016/j.cofs.2025.101370 | Recent review of NLP and text analysis in sensory practice. | Background |
| Chen, Gurdian, Sharma, Prinyawiwatkul and Torrico (2021), *Foods* 10(11): 2537, doi:10.3390/foods10112537 | Text mining and NLP for alternative-protein consumer studies. | Background |
| Torrico (2025), *Foods* 14(3): 464, doi:10.3390/foods14030464 | Case study prompting ChatGPT as a sensory evaluator; notes need for human-panel validation. | Background |

### LLMs as judges, annotation, semantic scoring

| Reference | Relevance | Role |
|---|---|---|
| Zheng, Chiang, Sheng et al. (2023), NeurIPS, arXiv:2306.05685 | Foundational LLM-as-judge paper; >80% LLM-human agreement and bias catalogue. | Background |
| Liu, Iter, Xu et al. (2023), G-Eval, EMNLP 2023: 2511–2522, doi:10.18653/v1/2023.emnlp-main.153 | LLM-based NLG evaluation correlating with humans in some settings; task-dependent. | Background |
| Gilardi, Alizadeh and Kubli (2023), *PNAS* 120(30): e2305016120, doi:10.1073/pnas.2305016120 | ChatGPT exceeded crowd workers by about 25 points on average at far lower cost on text-annotation tasks. | Findings |
| Gu et al. (2024/2026), survey on LLM-as-a-judge (circulates as arXiv:2411.15594 and as a journal version; verify venue/year) | Survey of biases, calibration, and human agreement; reliability is task-dependent. | Background |
| Motoki, Low and Velasco (2025), *Food Quality and Preference* 133: 105600, doi:10.1016/j.foodqual.2025.105600 | Framework for generative AI across concept, design, and testing phases in sensory and consumer research; flags bias, privacy, transparency. | Background |

### Semantic Similarity Rating

| Reference | Relevance | Role |
|---|---|---|
| Maier, Aslak, Fiaschi et al. (2025), arXiv:2510.08338, doi:10.48550/arXiv.2510.08338 | Defines SSR (free text mapped to a Likert PMF via embedding similarity to anchors). About 9,300 participants, 57 surveys, ~90% of human test-retest reliability. Distinct from direct single-score LLM rating. | Method |
| pymc-labs/semantic-similarity-rating (GitHub) | Reference implementation. | Method |

### Reasoning, chain-of-thought, and overthinking

| Reference | Relevance | Role |
|---|---|---|
| Wei et al. (2022), NeurIPS 35, doi:10.48550/arXiv.2201.11903 | Original chain-of-thought paper; the premise that added reasoning helps. | Background |
| Gema, Hägele, Chen et al. (2025), TMLR, doi:10.48550/arXiv.2507.14417 | Reports inverse scaling in test-time compute; failure modes include distraction, spurious-feature following, and overfitting to memorized framings. | Findings |
| Zhou, Ling, Chen, Wang, Fan and Wang (2026), "When More Thinking Hurts," arXiv:2604.10739 | Analyzes overthinking in test-time compute scaling and cases where extended reasoning abandons a correct answer. | Findings |
| Liu, Geng, Wu, Sucholutsky, Lombrozo and Griffiths (2024), "Mind Your Step (by Step)," arXiv:2410.21333 | Reports that chain-of-thought can reduce performance on tasks where thinking makes humans worse. | Findings |
| Sui, Chuang, Wang et al. (2025), "Stop Overthinking," TMLR accepted manuscript, doi:10.48550/arXiv.2503.16419 | Survey of efficient reasoning and overthinking. | Background |
| Kahneman (2011), *Thinking, Fast and Slow* | Dual-process metaphor; not a literal claim about model internals. | Background |
| Li et al. (2025), "From System 1 to System 2: a survey of reasoning large language models," arXiv:2502.17419 | Surveys application of dual-process framing to LLM reasoning modes. | Background |
| Hagendorff, Fabi and Kosinski (2022), "Thinking fast and slow in large language models," arXiv:2212.05206 | Reports System-1-like cognitive errors in LLMs. | Background |

### Tabular foundation models and baselines

| Reference | Relevance | Role |
|---|---|---|
| Hollmann, Müller, Purucker et al. (2025), *Nature* 637(8045): 319–326, doi:10.1038/s41586-024-08328-6 | TabPFN v2. Trained on >100M synthetic tasks from structural causal model priors; outperforms tuned ensembles on ≤10,000 samples with far less compute. | Method |
| Grinsztajn, Flöge et al. (2025), arXiv:2511.08667 | TabPFN-2.5, released November 2025. | Method |
| Hollmann, Müller, Eggensperger and Hutter (2022), arXiv:2207.01848 | Original TabPFN (classification only); establishes the prior-data-fitted-network approach. | Background |
| Qu, Holzmüller, Varoquaux and Le Morvan (2025), ICML, "TabICL" | Tabular foundation model for larger sample sizes. | Background |
| Grinsztajn, Oyallon and Varoquaux (2022), arXiv:2207.08815 | Reports that tree-based models often outperform deep learning on tabular data; benchmark context. | Background |
| Chen and Guestrin (2016), XGBoost; Prokhorenkova et al. (2018), CatBoost; Erickson et al. (2020), AutoGluon | Strong tree-ensemble and AutoML baselines in the small-to-medium tabular regime. | Background |

### LLM-based synthetic respondents (adjacent)

| Reference | Relevance | Role |
|---|---|---|
| Argyle et al. (2023), *Political Analysis* 31(3): 337–351, doi:10.1017/pan.2023.2 | "Silicon samples"; algorithmic fidelity of LLM-simulated respondents. | Background |
| Horton, Filippas and Manning (2023), NBER WP 31122, doi:10.3386/w31122 | LLMs as simulated economic agents. | Background |
| Brand, Israeli and Ngwe (2023), "Using LLMs for Market Research," HBS WP 23-062, SSRN 4395751 | Synthetic willingness-to-pay using LLMs. | Background |

## Reference verification status

The manuscript references were checked against the two verification files now stored in `paper/source_materials/`: `reference_verification_report.md` and `reference_verification.csv`. The manuscript bibliography is the authoritative reference list for this working paper. Remaining caution: the Gu et al. LLM-as-judge survey and several adjacent TabPFN or reasoning references appear only in this literature review and are not currently cited in the manuscript.

## References (BibTeX)

```
@article{mahieu2022ifc,
  title   = {Identifying drivers of liking and characterizing the ideal product thanks to {Free-Comment}},
  author  = {Mahieu, Benjamin and Visalli, Michel and Schlich, Pascal},
  journal = {Food Quality and Preference},
  volume  = {96},
  pages   = {104389},
  year    = {2022},
  doi     = {10.1016/j.foodqual.2021.104389}
}

@article{visalli2024hamdata,
  title   = {A dataset of perception and preferences of {French} consumers for commercial cooked hams sampled according to their nutritional values and claims},
  author  = {Visalli, Michel and Loiseau, Anne-Laure and Cordelle, Sylvie and Mahieu, Benjamin and Schlich, Pascal},
  journal = {Data in Brief},
  volume  = {54},
  pages   = {110549},
  year    = {2024},
  doi     = {10.1016/j.dib.2024.110549}
}

@article{mahieu2020fcwine,
  title   = {Free-comment outperformed check-all-that-apply in the sensory characterisation of wines with consumers at home},
  author  = {Mahieu, Benjamin and Visalli, Michel and Thomas, Arnaud and Schlich, Pascal},
  journal = {Food Quality and Preference},
  volume  = {84},
  pages   = {103937},
  year    = {2020},
  doi     = {10.1016/j.foodqual.2020.103937}
}

@article{mahieu2021mrca,
  title   = {A multiple-response chi-square framework for the analysis of {Free-Comment} and {Check-All-That-Apply} data},
  author  = {Mahieu, Benjamin and Schlich, Pascal and Visalli, Michel and Cardot, Herv{\'e}},
  journal = {Food Quality and Preference},
  volume  = {93},
  pages   = {104256},
  year    = {2021},
  doi     = {10.1016/j.foodqual.2021.104256}
}

@article{tenkleij2003freecomment,
  title   = {Text analysis of open-ended survey responses: a complementary method to preference mapping},
  author  = {ten Kleij, F. and Musters, P. A. D.},
  journal = {Food Quality and Preference},
  volume  = {14},
  number  = {1},
  pages   = {43--52},
  year    = {2003},
  doi     = {10.1016/S0950-3293(02)00011-3}
}

@article{symoneaux2012comments,
  title   = {Comment analysis of consumer's likes and dislikes as an alternative tool to preference mapping. A case study on apples},
  author  = {Symoneaux, R. and Galmarini, M. V. and Mehinagic, E.},
  journal = {Food Quality and Preference},
  volume  = {24},
  number  = {1},
  pages   = {59--66},
  year    = {2012},
  doi     = {10.1016/j.foodqual.2011.08.013}
}

@article{worch2013ipm,
  title   = {Ideal Profile Method (IPM): the ins and outs},
  author  = {Worch, Thierry and L{\^e}, S{\'e}bastien and Punter, Pieter and Pag{\`e}s, J{\'e}r{\^o}me},
  journal = {Food Quality and Preference},
  volume  = {28},
  number  = {1},
  pages   = {45--59},
  year    = {2013},
  doi     = {10.1016/j.foodqual.2012.08.001}
}

@article{hamilton2020nlp,
  title   = {Fast and automated sensory analysis: using natural language processing for descriptive lexicon development},
  author  = {Hamilton, Leah M. and Lahne, Jacob},
  journal = {Food Quality and Preference},
  volume  = {83},
  pages   = {103926},
  year    = {2020},
  doi     = {10.1016/j.foodqual.2020.103926}
}

@article{miller2021whisky,
  title   = {Sensory descriptor analysis of whisky lexicons through the use of deep learning},
  author  = {Miller, Chreston A. and Hamilton, Leah M. and Lahne, Jacob},
  journal = {Foods},
  volume  = {10},
  number  = {7},
  pages   = {1633},
  year    = {2021},
  doi     = {10.3390/foods10071633}
}

@misc{visallimahieu2023preprocessing,
  title   = {Combining statistics and semantic for an automated data analysis of Free-Comment sensory description of products},
  author  = {Visalli, Michel and Mahieu, Benjamin},
  howpublished = {15th Pangborn Sensory Science Symposium, Nantes, France},
  year    = {2023},
  note    = {HAL: hal-04197590}
}

@article{visalli2025madeleines,
  title   = {A dataset of annotated free comments on the sensory perception of madeleines for benchmarking text mining techniques},
  author  = {Visalli, Michel and Symoneaux, Ronan and Mursic, C{\'e}cile and Touret, Margaux and Lourtioux, Flore and Coulibaly, Kip{\'e}d{\`e}ne and Mahieu, Benjamin},
  journal = {Data in Brief},
  volume  = {58},
  pages   = {111250},
  year    = {2025},
  doi     = {10.1016/j.dib.2024.111250}
}

@article{hamilton2026review,
  title   = {Sensory applications of natural language processing and text analysis in practice: a review of recent literature},
  author  = {Hamilton, Kyle and Miller, Rebekah J. and Lahne, Jacob},
  journal = {Current Opinion in Food Science},
  volume  = {67},
  pages   = {101370},
  year    = {2026},
  doi     = {10.1016/j.cofs.2025.101370}
}

@article{chen2021textmining,
  title   = {Exploring text mining for recent consumer and sensory studies about alternative proteins},
  author  = {Chen, Ziyang and Gurdian, Cristhiam E. and Sharma, Chetan and Prinyawiwatkul, Witoon and Torrico, Damir D.},
  journal = {Foods},
  volume  = {10},
  number  = {11},
  pages   = {2537},
  year    = {2021},
  doi     = {10.3390/foods10112537}
}

@article{torrico2025chatgpt,
  title   = {The potential use of {ChatGPT} as a sensory evaluator of chocolate brownies: a brief case study},
  author  = {Torrico, Damir D.},
  journal = {Foods},
  volume  = {14},
  number  = {3},
  pages   = {464},
  year    = {2025},
  doi     = {10.3390/foods14030464}
}

@inproceedings{zheng2023judge,
  title     = {Judging {LLM-as-a-Judge} with {MT-Bench} and {Chatbot Arena}},
  author    = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric and Zhang, Hao and Gonzalez, Joseph and Stoica, Ion},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
  eprint    = {2306.05685}
}

@inproceedings{liu2023geval,
  title   = {{G-Eval}: {NLG} evaluation using {GPT-4} with better human alignment},
  author  = {Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and Xu, Ruochen and Zhu, Chenguang},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages   = {2511--2522},
  year    = {2023},
  doi     = {10.18653/v1/2023.emnlp-main.153}
}

@article{gilardi2023chatgpt,
  title   = {{ChatGPT} outperforms crowd workers for text-annotation tasks},
  author  = {Gilardi, Fabrizio and Alizadeh, Meysam and Kubli, Ma{\"e}l},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {120},
  number  = {30},
  pages   = {e2305016120},
  year    = {2023},
  doi     = {10.1073/pnas.2305016120}
}

@misc{gu2024judgesurvey,
  title   = {A survey on {LLM-as-a-judge}},
  author  = {Gu, Jiawei and others},
  year    = {2024},
  eprint  = {2411.15594},
  archivePrefix = {arXiv},
  note    = {also circulates in a journal version; verify venue and year}
}

@article{motoki2025genai,
  title   = {Generative {AI} framework for sensory and consumer research},
  author  = {Motoki, Kosuke and Low, Julia and Velasco, Carlos},
  journal = {Food Quality and Preference},
  volume  = {133},
  pages   = {105600},
  year    = {2025},
  doi     = {10.1016/j.foodqual.2025.105600}
}

@misc{maier2025ssr,
  title   = {{LLMs} reproduce human purchase intent via semantic similarity elicitation of {Likert} ratings},
  author  = {Maier, Benjamin F. and Aslak, Ulf and Fiaschi, Luca and Rismal, Nina and Fletcher, Kemble and Luhmann, Christian C. and Dow, Robbie and Pappas, Kli and Wiecki, Thomas V.},
  year    = {2025},
  eprint  = {2510.08338},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2510.08338}
}

@inproceedings{wei2022cot,
  title     = {Chain-of-thought prompting elicits reasoning in large language models},
  author    = {Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Ichter, Brian and Xia, Fei and Chi, Ed and Le, Quoc V. and Zhou, Denny},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  eprint    = {2201.11903},
  doi       = {10.48550/arXiv.2201.11903}
}

@misc{gema2025inversescaling,
  title   = {Inverse scaling in test-time compute},
  author  = {Gema, Aryo Pradipta and H{\"a}gele, Alexander and Chen, Runjin and Arditi, Andy and Goldman-Wetzler, Jacob and Fraser-Taliente, Kit and Sleight, Henry and Petrini, Linda and Michael, Julian and Alex, Beatrice and Minervini, Pasquale and Chen, Yanda and Benton, Joe and Perez, Ethan},
  year    = {2025},
  eprint  = {2507.14417},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2507.14417},
  note    = {published in TMLR, 12/2025}
}

@misc{zhou2026overthinking,
  title   = {When more thinking hurts: overthinking in {LLM} test-time compute scaling},
  author  = {Zhou, Shu and Ling, Rui and Chen, Junan and Wang, Xin and Fan, Tao and Wang, Hao},
  year    = {2026},
  eprint  = {2604.10739},
  archivePrefix = {arXiv}
}

@misc{liu2024mindyourstep,
  title   = {Mind your step (by step): chain-of-thought can reduce performance on tasks where thinking makes humans worse},
  author  = {Liu, Ryan and Geng, Jiayi and Wu, Addison J. and Sucholutsky, Ilia and Lombrozo, Tania and Griffiths, Thomas L.},
  year    = {2024},
  eprint  = {2410.21333},
  archivePrefix = {arXiv}
}

@article{hollmann2025tabpfn,
  title   = {Accurate predictions on small data with a tabular foundation model},
  author  = {Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and Schirrmeister, Robin Tibor and Hutter, Frank},
  journal = {Nature},
  volume  = {637},
  number  = {8045},
  pages   = {319--326},
  year    = {2025},
  doi     = {10.1038/s41586-024-08328-6}
}

@misc{grinsztajn2025tabpfn25,
  title   = {{TabPFN-2.5}: advancing the state of the art in tabular foundation models},
  author  = {Grinsztajn, L{\'e}o and Fl{\"o}ge, Klemens and others},
  year    = {2025},
  eprint  = {2511.08667},
  archivePrefix = {arXiv}
}

@article{argyle2023outofonemany,
  title   = {Out of one, many: using language models to simulate human samples},
  author  = {Argyle, Lisa P. and Busby, Ethan C. and Fulda, Nancy and Gubler, Joshua R. and Rytting, Christopher and Wingate, David},
  journal = {Political Analysis},
  volume  = {31},
  number  = {3},
  pages   = {337--351},
  year    = {2023},
  doi     = {10.1017/pan.2023.2}
}

@techreport{horton2023homosilicus,
  title       = {Large language models as simulated economic agents: what can we learn from {Homo Silicus}?},
  author      = {Horton, John J. and Filippas, Apostolos and Manning, Benjamin S.},
  institution = {National Bureau of Economic Research},
  number      = {31122},
  year        = {2023},
  doi         = {10.3386/w31122}
}

@book{kahneman2011thinking,
  title     = {Thinking, Fast and Slow},
  author    = {Kahneman, Daniel},
  publisher = {Farrar, Straus and Giroux},
  year      = {2011},
  isbn      = {9780374533557}
}

@article{wang2025chemosensory,
  title   = {Automating chemosensory creativity assessment with large language models},
  author  = {Wang, Qian Janice and Pellegrino, Robert},
  journal = {Food Quality and Preference},
  volume  = {132},
  pages   = {105599},
  year    = {2025},
  doi     = {10.1016/j.foodqual.2025.105599}
}

@article{jakkaew2026coffee,
  title   = {{LLM}-driven annotation: a scalable framework for automated multi-label coffee flavor classification},
  author  = {Jakkaew, Prasittichai and Chondamrongkul, Nacha and Suebsombut, Paweena},
  journal = {ECTI Transactions on Computer and Information Technology},
  volume  = {20},
  number  = {2},
  pages   = {367--382},
  year    = {2026},
  doi     = {10.37936/ecti-cit.2026202.264386},
  note    = {confirm author initials and details at source}
}

@misc{li2025system1to2,
  title   = {From System 1 to System 2: a survey of reasoning large language models},
  author  = {Li, Zhong-Zhi and others},
  year    = {2025},
  eprint  = {2502.17419},
  archivePrefix = {arXiv},
  note    = {confirm full author list at source}
}

@misc{hagendorff2022thinkingfast,
  title   = {Thinking fast and slow in large language models},
  author  = {Hagendorff, Thilo and Fabi, Sarah and Kosinski, Michal},
  year    = {2022},
  eprint  = {2212.05206},
  archivePrefix = {arXiv}
}

@misc{hollmann2022tabpfn,
  title   = {{TabPFN}: a transformer that solves small tabular classification problems in a second},
  author  = {Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  year    = {2022},
  eprint  = {2207.01848},
  archivePrefix = {arXiv}
}

@misc{grinsztajn2022trees,
  title   = {Why do tree-based models still outperform deep learning on typical tabular data?},
  author  = {Grinsztajn, L{\'e}o and Oyallon, Edouard and Varoquaux, Ga{\"e}l},
  year    = {2022},
  eprint  = {2207.08815},
  archivePrefix = {arXiv}
}
```
