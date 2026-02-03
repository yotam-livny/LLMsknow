# LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

**Authors:** Hadas Orgad¹, Michael Toker¹, Zorik Gekhman¹, Roi Reichart¹, Idan Szpektor², Hadas Kotek³, Yonatan Belinkov¹  
**Affiliations:** ¹Technion, ²Google Research, ³Apple  
**Conference:** ICLR 2025  
**arXiv:** [2410.02707](https://arxiv.org/abs/2410.02707)

---

## Abstract

Large language models (LLMs) often produce errors, including factual inaccuracies, biases, and reasoning failures, collectively referred to as "hallucinations". Recent studies have demonstrated that LLMs' internal states encode information regarding the truthfulness of their outputs, and that this information can be utilized to detect errors. 

This work shows that the internal representations of LLMs encode **much more information about truthfulness than previously recognized**. The paper demonstrates that:

1. Truthfulness information is concentrated in specific tokens
2. Error detectors fail to generalize across datasets (truthfulness encoding is multifaceted, not universal)
3. Internal representations can predict the types of errors the model is likely to make
4. There's a discrepancy between LLMs' internal encoding and external behavior: they may encode the correct answer yet consistently generate an incorrect one

These insights deepen our understanding of LLM errors from the model's internal perspective, which can guide future research on enhancing error analysis and mitigation.

---

## 1. Introduction

### Problem Statement

The growing popularity of LLMs has highlighted their tendency to "hallucinate" - generating inaccurate information. While previous research has:

- **Extrinsic analysis**: Examined how users perceive errors (behavioral analysis)
- **Intrinsic analysis**: Explored internal representations for error detection

However, these approaches were typically restricted to detecting errors without delving deeper into:
- How signals are represented
- How they could be leveraged to understand or mitigate hallucinations
- The types of information encoded

### Scope

The paper adopts a **broad interpretation of hallucinations**, considering them to encompass:
- Factual inaccuracies
- Biases
- Common-sense reasoning failures
- Other real-world errors

This enables drawing general conclusions about model errors from a broad perspective.

---

## 2. Key Findings

### Finding 1: Truthfulness Information is Concentrated in Specific Tokens

**Discovery**: The choice of token used to extract truthfulness signals is crucial. Previous studies overlooked this nuance.

**Key Insight**: Truthfulness information is concentrated in the **exact answer tokens** - e.g., "Hartford" in "The capital of Connecticut is Hartford, an iconic city...".

**Impact**: Recognizing this significantly improves error detection strategies across the board.

### Finding 2: Error Detectors Don't Generalize Across Datasets

**Discovery**: Error detectors trained on one dataset fail to generalize to others.

**Implication**: Contrary to prior claims, **truthfulness encoding is not universal but rather multifaceted**. Different types of errors are encoded differently in the model's internal representations.

### Finding 3: Predicting Error Types

**Discovery**: Internal representations can be used to predict the **types of errors** the model is likely to make.

**Application**: This facilitates the development of tailored mitigation strategies based on error type classification.

### Finding 4: Discrepancy Between Internal Encoding and External Behavior

**Discovery**: LLMs may encode the correct answer internally, yet consistently generate an incorrect one externally.

**Implication**: This reveals a fundamental disconnect between what the model "knows" internally and what it outputs, suggesting opportunities for intervention.

---

## 3. Methodology

### Experimental Framework

The paper trains classifiers on internal representations to predict various features related to the truthfulness of generated outputs. Experiments cover a broad array of LLM limitations.

### Datasets Used

Based on the codebase, the paper evaluates on:

1. **TriviaQA** - Question answering
2. **IMDB** - Sentiment classification
3. **Winobias** - Bias detection
4. **Winogrande** - Commonsense reasoning
5. **HotpotQA** - Multi-hop question answering
6. **Math** - Mathematical reasoning
7. **Movies** - Movie-related QA
8. **MNLI** - Natural language inference
9. **Natural Questions** - Open-domain QA

### Models Evaluated

- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mistral-7B-v0.3`
- `meta-llama/Meta-Llama-3-8B`
- `meta-llama/Meta-Llama-3-8B-Instruct`

### Probing Locations

The paper probes internal representations at various locations:
- **MLP layers** (Multi-Layer Perceptron)
- **MLP last layer only**
- **MLP last layer input**
- **Attention output**

### Token Selection Strategy

Key tokens probed include:
- `last_q_token` - Last token of the question
- `first_answer_token` - First token of the answer
- `exact_answer_first_token` - First token of the exact answer
- `exact_answer_last_token` - Last token of the exact answer
- `exact_answer_after_last_token` - Token after the exact answer
- Relative positions: `-8, -7, -6, -5, -4, -3, -2, -1` (from end)

---

## 4. Results Summary

### Error Detection Performance

The paper demonstrates significant improvements in error detection by:
- Focusing on exact answer tokens
- Using appropriate layer and token combinations
- Training dataset-specific classifiers

### Error Type Classification

The paper identifies and classifies different error types:

#### (A) Refuses to Answer
- Model refuses to provide an answer

#### (B) Consistently Correct
- **(B1) All**: All resamples are correct
- **(B2) Most**: Most resamples are correct

#### (C) Consistently Incorrect
- **(C1) All**: All resamples are incorrect
- **(C2) Most**: Most resamples are incorrect

#### (D) Two Competing Answers
- Model oscillates between two different answers

#### (E) Many Different Answers
- **(E1) Non-correct**: Many different incorrect answers
- **(E2) Correct appears**: Many answers, but correct one appears sometimes

### Answer Choice Strategies

The paper evaluates different strategies for selecting answers from multiple resamples:

1. **Greedy**: Original greedy decoding
2. **Random**: Random selection from valid answers
3. **Majority**: Select most frequent answer
4. **Probing**: Use internal representations to predict correctness

**Key Result**: Probing-based selection outperforms other strategies across error types, especially for:
- Consistently incorrect cases (C2)
- Two competing answers (D)
- Many answers where correct appears (E2)

### Performance by Error Type

From the results tables:

**Mistral-7B-Instruct on TriviaQA:**
- All errors: Greedy 0.63 → Probing 0.71
- Consistently incorrect (C2): Greedy 0.11 → Probing 0.53
- Two competing (D): Greedy 0.32 → Probing 0.78
- Many answers, correct appears (E2): Greedy 0.23 → Probing 0.56

**Llama-3-8B-Instruct on TriviaQA:**
- All errors: Greedy 0.69 → Probing 0.73
- Consistently incorrect (C2): Greedy 0.12 → Probing 0.43
- Two competing (D): Greedy 0.43 → Probing 0.60
- Many answers, correct appears (E2): Greedy 0.28 → Probing 0.52

---

## 5. Implications and Contributions

### Theoretical Contributions

1. **Token-specific encoding**: Truthfulness information is concentrated in specific tokens, not uniformly distributed
2. **Multifaceted encoding**: Truthfulness encoding is dataset/error-type specific, not universal
3. **Internal-external disconnect**: Models can encode correct information internally while generating incorrect outputs

### Practical Contributions

1. **Improved error detection**: Better strategies for identifying errors in LLM outputs
2. **Error type prediction**: Ability to predict what types of errors will occur
3. **Answer selection**: Using internal representations to select better answers from multiple generations
4. **Tailored mitigation**: Different error types may require different mitigation strategies

### Limitations

1. **Lack of generalization**: Error detectors don't generalize across datasets
2. **Dataset-specific**: Requires training on each dataset separately
3. **Computational cost**: Requires multiple resamples and probing

---

## 6. Future Directions

The paper suggests several directions for future research:

1. **Universal error detection**: Developing methods that generalize across datasets
2. **Mitigation strategies**: Using error type predictions to develop targeted interventions
3. **Training improvements**: Leveraging internal representations to improve model training
4. **Interpretability**: Better understanding of how and why models encode truthfulness information

---

## 7. Technical Details

### Probing Methodology

1. **Extract internal representations** at specific layers and tokens
2. **Train classifiers** (Logistic Regression) to predict correctness
3. **Evaluate** on validation/test sets
4. **Compare** with baselines (logprob, p_true)

### Resampling Strategy

- Generate multiple answers using sampling (temperature, top_p)
- Extract exact answers from each resample
- Classify error types based on correctness patterns
- Use probing to select best answer

### Baselines Compared

1. **Logprob detection**: Using log probabilities of generated tokens
2. **P_true detection**: Asking model to evaluate truthfulness of its own outputs
3. **Random selection**: Random choice from valid answers
4. **Majority voting**: Most frequent answer

---

## 8. Conclusion

This work reveals that LLMs' internal representations encode much more information about truthfulness than previously recognized. The key insights are:

1. **Token concentration**: Truthfulness signals are concentrated in specific tokens
2. **Multifaceted encoding**: Different error types are encoded differently
3. **Error type prediction**: Internal representations can predict error types
4. **Internal-external gap**: Models may know the correct answer but not generate it

These findings deepen our understanding of LLM errors from an internal perspective and provide a foundation for:
- Better error detection
- Error type classification
- Targeted mitigation strategies
- Improved answer selection

The work bridges intrinsic (internal) and extrinsic (behavioral) analysis of LLM errors, providing a more complete picture of how and why hallucinations occur.

---

## References

- **Paper**: [arXiv:2410.02707](https://arxiv.org/abs/2410.02707)
- **Website**: [llms-know.github.io](https://llms-know.github.io/)
- **Code**: Available in this repository

---

## Notes

This summary is based on the paper content and the codebase structure. For complete details, methodology, and full results, please refer to the original paper and supplementary materials.
