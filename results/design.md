# Design frozen before observing edit results

Model: Qwen2.5-1.5B-Instruct, BF16, chat format with a fixed concise-answer system instruction.
Edits: 2+2 -> 5; 2+2 -> 6; 3+4 -> 8. Each run starts from the same unedited model. Seeds 0,1,2.

Scope A (string exception): only the exact user message should change.
Scope B (computed-fact rewrite): alternate expressions of the same sum and explicit downstream uses should change. This is an operational rewrite of a designated inner expression, not a consistent alternative arithmetic theory.

Editors: an exact-string output wrapper; last-MLP rank-8 adapter with exact-prompt CE and KL coefficients 0,1,10,100; same adapter with preservation of three nonexact training variants (coefficients 10,100); four-prompt paraphrase training with coefficients 1,10,100. All trained 80 steps; checkpoints at 20 and 80. Same optimizer and adapter location throughout. This compares training objectives, not state-of-the-art editing algorithms.

Primary evaluation: unrestricted-vocabulary next-token argmax, full-vocabulary KL and total variation. Numeric answer success only where the true and desired answer are each one token. Multi-digit arithmetic uses first-token drift only; it must not be described as complete-answer accuracy. Full greedy continuation audit will examine representative configurations separately.

Test bank: held-out prompt templates, small sum grid, subtraction/multiplication, large sums, and CounterFact prompts. Locality training uses distinct prompts and disjoint CounterFact records. Three boundary-preservation prompts are training data, never paraphrase test data. No test-based checkpoint selection: report both fixed checkpoints, with step 80 primary.

Interpretation limits: one model, one edited layer, three edit requests but only two underlying sums; seeds are optimization repetitions, not independent knowledge samples. No universal isolation certificate; no claim to measure subjective belief or identify arithmetic helices.

## Explicitly exploratory follow-ups

After inspecting the final-layer results, added 18 middle-layer (block 14) runs: exact and paraphrase objectives, lambda=1, three edits, three seeds, otherwise matched optimizer/rank/steps. These use full forward/backward passes, not final-layer caching. They were motivated by the absence of base-correct propagation in the primary final-layer settings.

After observing middle-layer propagation, added matched dependent controls: replace inner 2+2 with 1+3, and inner 3+4 with 2+5. Both retain the ordinary sum, but neither contains the edited operand pair. Evaluate every final adapter on all 18 corresponding control prompts; report accidental counterfactual adoption on base-correct controls separately. This tests specificity of the apparent downstream transfer.

Generation implementation corrections: the model's default repetition penalty was initially inherited (1.1); primary audit explicitly disables it (1.0). The initial outputs are retained separately. Cross-batch BF16 numerical variation is assessed with same-batch full-model/cache checks; all primary complete-answer results come from full-model generation. These corrections do not change the training sweep.
