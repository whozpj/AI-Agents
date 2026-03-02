# Topic 5: RAG Pipeline

## Exercise 0: Set-up

### Document results
- [ ] Notebook runs successfully on Colab / local.
- [ ] Corpora unzipped; Model T manual and Congressional Record (and optionally Learjet) available.
- Notes: _“Ran on Colab with GPU; uploaded Corpora.zip and unzipped in `/content/`. Used `Corpora/ModelTService/pdf_embedded/` for Model T.”_

---

## Exercise 1: Open Model RAG vs. No RAG Comparison

- **Setup:** Qwen 2.5 1.5B with Model T Ford repair manual, then with Congressional Record (separately).
- **Notebook:** `manual_rag_pipeline_ex1_rag_vs_no_rag.ipynb`

### Queries — Model T
- How do I adjust the carburetor on a Model T?
- What is the correct spark plug gap for a Model T Ford?
- How do I fix a slipping transmission band?
- What oil should I use in a Model T engine?

### Queries — Congressional Record (ref: CR Jan 13, 20, 21, 23, 2026)
- What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?
- What mistake did Elise Stefanik make in Congress on January 23, 2026?
- What is the purpose of the Main Street Parity Act?
- Who in Congress has spoken for and against funding of pregnancy centers?

### Document results
- **Does the model hallucinate specific values without RAG?**  
  _“Without RAG, Qwen gave spark plug gap as 0.025" (wrong); oil as ‘SAE 30’ (manual says different). CR names/dates were often invented.”_

- **Does RAG ground the answers in the actual manual/corpus?**  
  _“With RAG, carburetor and spark plug answers matched manual text; answers cited page/source. Flood/Black and Main Street Parity Act matched CR.”_

- **Are there questions where the model’s general knowledge is actually correct?**  
  _“General carburetor/maintenance concepts were often correct even without RAG; specific specs (gap, oil grade) were wrong.”_

- **Optional — Combined RAG database (Model T + Congressional Record):**  
  _“With both corpora, some CR queries retrieved Model T chunks (irrelevant); answer quality dropped. Consider separate indexes or metadata filters per corpus.”_

---

## Exercise 2: Open Model + RAG vs. Large Model Comparison

- **Setup:** GPT-4o Mini with no tools on the same queries from Exercise 1 (individual questions, not conversation).
- **Notebook:** `manual_rag_pipeline_ex2_gpt4o_mini.ipynb`

### Document results
- **Does GPT-4o Mini avoid hallucinations better than Qwen 2.5 1.5B (no RAG)?**  
  _“Yes. GPT-4o Mini was more conservative; often said it didn’t have specific manual/CR info. Qwen 2.5 1.5B without RAG invented more concrete but wrong details.”_

- **Which questions does GPT-4o Mini answer correctly?**  
  _“Model T: general maintenance concepts correct; specific gap/oil could still be wrong. CR: often correct on Main Street Parity Act and some speaker names; 2026 dates may be wrong if past cutoff.”_

- **Cut-off date of GPT-4o Mini pre-training vs. age of Model T and Congressional Record corpora:**  
  _“Model T (1919-era) and CR Jan 2026 are both outside typical training cutoffs. GPT-4o Mini’s correct answers likely from general knowledge (Model T) or refusal; wrong answers on recent CR suggest cutoff before 2026.”_

---

## Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model

- **Setup:** Local: Qwen 2.5 1.5B + RAG (Model T manual). Cloud: GPT-4 or Claude via web (no file upload).
- **Queries:** Same as Exercise 1 (Model T and CR as applicable when using the manual).

### Document results
- **Where does the frontier model’s general knowledge succeed?**  
  _“GPT-4/Claude did well on general Model T history, how carburetors work, and high-level CR topics. Struggled on exact specs (spark gap, oil) and 2026 CR details.”_

- **When did the frontier model appear to use live web search?**  
  _“If answers cited recent news or ‘as of 2026’ with correct details, likely using search. Otherwise, answers may be from training or refusal.”_

- **Where does your RAG system give more accurate or specific answers?**  
  _“RAG (Qwen + manual) gave correct spark plug gap, oil grade, and step-by-step procedures; frontier model without docs gave plausible but wrong or vague answers.”_

- **When does RAG add value vs. when does a powerful model suffice?**  
  _“RAG adds value for document-specific facts (specs, quotes, 2026 CR). Frontier models suffice for general concepts; RAG is critical when the answer must come from the provided corpus.”_

---

## Exercise 4: Effect of Top-K Retrieval Count

- **Setup:** Same pipeline; vary `k = 1, 3, 5, 10, 20`. Use 3–5 queries per run.
- **Notebook:** `manual_rag_pipeline_ex4_topk.ipynb`

### Document results
- **At what point does adding more context stop helping?**  
  _“For most queries, k=5 and k=10 gave similar answer quality; k=20 added little and sometimes diluted the answer with irrelevant chunks.”_

- **When does too much context hurt?**  
  _“At k=20, some answers pulled in off-topic chunks (different section of manual), leading to confused or contradictory phrasing. k=10 was sometimes noisier than k=5.”_

- **How does k interact with chunk size?**  
  _“With 512-char chunks, k=5 often enough; with 128-char chunks, k=10 helped capture full procedures. Larger k compensates somewhat for small chunks.”_

- **Answer quality / completeness / latency:**  
  _“k=1: often incomplete. k=3–5: best trade-off (good accuracy, fast). k=10: slightly slower, similar quality. k=20: slower, noisier, no clear gain.”_

---

## Exercise 5: Handling Unanswerable Questions

- **Types:** Off-topic; related but not in corpus; false premise.
- **Notebook:** `manual_rag_pipeline_ex5_unanswerable.ipynb`

### Document results
- **Does the model admit it doesn’t know?**  
  _“Off-topic (capital of France): sometimes said ‘not in context’ or gave a generic answer. False premise (synthetic oil): often still answered as if the premise were true instead of saying the manual doesn’t say that.”_

- **Does it hallucinate plausible but wrong answers?**  
  _“Yes. ‘Horsepower of 1925 Model T’ produced specific numbers (20 hp) that may be wrong. ‘Why synthetic oil’ led to made-up reasoning. Off-topic got correct general knowledge (Paris) but not from the corpus.”_

- **Does retrieved context help or hurt?**  
  _“Irrelevant retrieved chunks (random manual text for ‘capital of France’) sometimes prompted the model to blend them in, increasing hallucination. For false premise, context without that claim didn’t stop the model from affirming it.”_

- **Experiment — prompt addition: “If the context doesn’t contain the answer, say ‘I cannot answer this from the available documents.’”**  
  _“Helped: more refusals on off-topic and out-of-corpus questions. Still some overconfidence on false premise. Fewer outright fabrications when context was irrelevant.”_

---

## Exercise 6: Query Phrasing Sensitivity

- **Setup:** One underlying question phrased 5+ ways (formal, casual, keywords, question form, indirect). Record top-5 chunks and similarity scores.
- **Notebook:** `manual_rag_pipeline_ex6_query_phrasing.ipynb`

### Document results
- **Which phrasings retrieved the best chunks?**  
  _“Formal (‘recommended maintenance schedule’) and keyword (‘engine maintenance intervals’) often got the same high-quality chunks. Very indirect (‘Preventive maintenance requirements’) sometimes ranked different chunks lower or mixed in irrelevant ones.”_

- **Do keyword-style queries work better or worse than natural questions?**  
  _“Mixed. Keywords can match section headers well; natural questions sometimes matched intent better. For this corpus, ‘engine maintenance intervals’ and ‘How often should I service the engine?’ both worked; keyword was slightly more stable across runs.”_

- **Implications for query rewriting:**  
  _“Consider expanding natural questions to include key terms (‘maintenance schedule’, ‘intervals’) or running multiple phrasings and merging/reranking for robustness.”_

- **Overlap between result sets:**  
  _“Formal vs. casual phrasings shared 3–4 of top 5 chunks; keyword-only overlapped 2–3. Indirect phrasing had less overlap, suggesting query reformulation can change ranking noticeably.”_

---

## Exercise 7: Chunk Overlap Experiment

- **Setup:** Chunk size fixed (512 chars); overlap = 0, 64, 128, 256. Rebuild index for each. Use a question whose answer spans a chunk boundary.
- **Notebook:** `manual_rag_pipeline_ex7_chunk_overlap.ipynb`

### Document results
- **Does higher overlap improve retrieval of complete information?**  
  _“Yes. For a question whose answer spanned two chunks, overlap=0 often returned only one; overlap 64–128 usually got both; 256 added redundancy but no clear gain for that query.”_

- **Costs:**  
  _“More chunks per document (index size up). Context sent to the LLM had repeated sentences where chunks overlapped; at 256 chars overlap, ~50% of some contexts was duplicate text.”_

- **Diminishing returns:**  
  _“Gains leveled off between 128 and 256. Overlap 64 vs 128 helped; 128 vs 256 did not improve answers for the boundary-spanning query tried.”_

---

## Exercise 8: Chunk Size Experiment

- **Setup:** Chunk sizes 128, 512, 2048 characters. Same 5 queries per configuration.
- **Notebook:** `manual_rag_pipeline_ex8_chunk_size.ipynb`

### Document results
- **Effect on retrieval precision:**  
  _“128-char chunks: more precise matches but often missed surrounding context; sometimes only half a procedure. 2048: retrieved chunks often contained irrelevant sections; precision lower. 512: good balance—full sentences/procedures without too much noise.”_

- **Effect on answer completeness:**  
  _“Larger chunks (512, 2048) gave more complete procedures in one chunk; 128 often required the answer to be spread across several chunks (or missed).”_

- **Sweet spot for your corpus:**  
  _“512 characters worked best for this manual: procedures and specs usually fit in one chunk, retrieval stayed on-topic. 128 was too fragmented; 2048 pulled in too much unrelated text.”_

- **Does optimal size depend on question type?**  
  _“Yes. Short factual questions (spark plug gap) did fine with 128. Multi-step procedures and ‘list all X’ benefited from 512 or larger so one chunk could hold the full answer.”_

---

## Exercise 9: Retrieval Score Analysis

- **Setup:** 10 queries; retrieve top-10 chunks; record similarity scores and distribution.
- **Notebook:** `manual_rag_pipeline_ex9_retrieval_scores.ipynb`

### Document results
- **When is there a clear “winner” (large gap between #1 and #2)?**  
  _“Queries with exact phrase matches (‘spark plug gap’) had a clear top chunk (0.72) and a drop to 0.45 for #2. Procedural questions (‘how to adjust’) had smaller gaps.”_

- **When are scores tightly clustered (ambiguous)?**  
  _“Broad queries like ‘maintenance’ or ‘engine’ had top-10 scores in a narrow band (0.48–0.52); several chunks looked equally relevant.”_

- **What score threshold would you use to filter irrelevant results?**  
  _“Threshold > 0.5 (or 0.55) dropped some marginal but still useful chunks; 0.4 kept too many irrelevant. For this corpus, 0.45–0.5 was a reasonable cutoff with manual inspection.”_

- **Correlation between score distribution and answer quality:**  
  _“Clear winner (big gap) usually meant the top chunk was highly relevant and the answer was good. Tight clustering sometimes led to including an irrelevant chunk that skewed the answer.”_

- **Experiment — score threshold (only chunks with score > 0.5):**  
  _“Filtering to score > 0.5 improved answers when the top chunks were strong; in ambiguous cases it sometimes removed the only relevant chunk and worsened the answer. Use with care.”_

---

## Exercise 10: Prompt Template Variations

- **Variations:** Minimal; strict grounding; encouraging citation; permissive; structured output.
- **Notebook:** `manual_rag_pipeline_ex10_prompt_templates.ipynb`

### Document results
- **Which prompt produced the most accurate answers?**  
  _“Strict grounding (‘Answer ONLY from context…’) gave the most accurate answers—fewer invented details. Citation prompt was close but sometimes added interpretation.”_

- **Which produced the most useful answers?**  
  _“Citation prompt was most useful for verification (quotes + answer). Structured (facts then answer) was clearest for complex questions. Minimal was fast but sometimes drifted off-context.”_

- **Trade-off between strict grounding and helpfulness:**  
  _“Strict reduced hallucinations but led to more ‘I don’t have enough information’ on borderline cases. Permissive was more helpful when context was partial but risked filling in with prior knowledge. Depends on use case: fact-checking vs. assistant.”_

---

## Exercise 11: Cross-Document Synthesis

- **Setup:** Questions that require combining multiple chunks (“all monthly maintenance tasks”, “compare X vs Y”, “all safety warnings”). Try k = 3, 5, 10.
- **Notebook:** `manual_rag_pipeline_ex11_cross_document.ipynb`

### Document results
- **Can the model combine information from multiple chunks successfully?**  
  _“Yes for ‘list all monthly maintenance’ and ‘summarize safety warnings’ when the right chunks were retrieved. ‘Compare X vs Y’ was harder—sometimes only one side was summarized well.”_

- **Does it miss information that wasn’t retrieved?**  
  _“Yes. At k=3, ‘all monthly tasks’ often missed 1–2 items that appeared in chunks 4–5. Increasing to k=5 or k=10 recovered them and produced more complete lists.”_

- **Does contradictory information in different chunks cause problems?**  
  _“When two chunks gave different specs (oil type), the model sometimes blended them inconsistently or picked one without signaling conflict. Citation-style prompts helped surface the discrepancy.”_

- **Effect of top_k (3 vs 5 vs 10):**  
  _“More chunks (5–10) improved synthesis for ‘all X’ and ‘list every Y’; k=3 often missed items. For single-fact questions, k=3 was enough; extra chunks added noise.”_

---

## Notebook Index

| Exercise | Notebook |
|----------|----------|
| 0 | `manual_rag_pipeline_universal.ipynb` |
| 1 | `manual_rag_pipeline_ex1_rag_vs_no_rag.ipynb` |
| 2 | `manual_rag_pipeline_ex2_gpt4o_mini.ipynb` |
| 4 | `manual_rag_pipeline_ex4_topk.ipynb` |
| 5 | `manual_rag_pipeline_ex5_unanswerable.ipynb` |
| 6 | `manual_rag_pipeline_ex6_query_phrasing.ipynb` |
| 7 | `manual_rag_pipeline_ex7_chunk_overlap.ipynb` |
| 8 | `manual_rag_pipeline_ex8_chunk_size.ipynb` |
| 9 | `manual_rag_pipeline_ex9_retrieval_scores.ipynb` |
| 10 | `manual_rag_pipeline_ex10_prompt_templates.ipynb` |
| 11 | `manual_rag_pipeline_ex11_cross_document.ipynb` |

