# Tinker experiment

This folder prepares the specialization experiment without claiming that training has already been performed.

## Goal

Teach a supported open-weight model to produce:

- concise outdoor missions
- direct real-world observation
- exactly two useful questions
- a safety instruction
- a short reflection prompt

## Important

Before training:

1. Confirm the model is currently supported by your Tinker account.
2. Record the exact model ID.
3. Keep the evaluation set held out.
4. Do not train on evaluation prompts.
5. Save the exact configuration and checkpoint identifier.
6. Compare the tuned model against an untuned baseline.

## Data

`dataset/train.jsonl` contains starter examples.

Expand it manually to at least 80 high-quality examples before a serious run. Prefer quality over duplicated templates.

`dataset/eval.jsonl` must remain held out.

## Recommended evidence for the DEV article

Publish:

- base model
- number of training examples
- number of held-out prompts
- training method
- tuned checkpoint identifier
- baseline score
- tuned score
- latency comparison
- limitations and failures

Do not invent performance gains.
