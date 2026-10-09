import json
from pathlib import Path

import torch
import tinker
from tinker import TensorData

BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 16
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 1

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "train_messages.jsonl"


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_example(row):
    messages = row["messages"]

    system = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"]

    prompt = (
        f"System:\n{system}\n\n"
        f"User:\n{user}\n\n"
        f"Assistant:\n"
    )

    return prompt, assistant


def build_datum(tokenizer, row):
    prompt, answer = format_example(row)

    prompt_tokens = tokenizer.encode(prompt)
    answer_tokens = tokenizer.encode(answer)

    all_tokens = prompt_tokens + answer_tokens

    model_input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]

    # Do not train on prompt tokens.
    weights = [0.0] * max(len(prompt_tokens) - 1, 0)
    weights += [1.0] * len(answer_tokens)

    # Keep lengths aligned exactly.
    weights = weights[:len(target_tokens)]

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.long)
            ),
            "weights": TensorData.from_torch(
                torch.tensor(weights, dtype=torch.float32)
            ),
        },
    )


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    rows = load_rows()

    print(f"Training examples: {len(rows)}")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    service = tinker.ServiceClient()

    training_client = service.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        user_metadata={
            "project": "TrailTutor-AI",
            "experiment": "sft-v1"
        }
    )

    tokenizer = training_client.get_tokenizer()

    datums = [build_datum(tokenizer, row) for row in rows]

    step = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for batch in batches(datums, BATCH_SIZE):
            step += 1

            fb_future = training_client.forward_backward(
                batch,
                loss_fn="cross_entropy"
            )

            optim_future = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=LEARNING_RATE
                )
            )

            fb_result = fb_future.result()
            optim_future.result()

            loss = fb_result.metrics.get("loss:sum")

            print(
                f"step={step:03d} "
                f"batch={len(batch)} "
                f"loss={loss}"
            )

    print("\nSaving training state...")

    state = training_client.save_state(
        "trailtutor-sft-v1",
        user_metadata={
            "examples": str(len(rows)),
            "epochs": str(EPOCHS),
            "batch_size": str(BATCH_SIZE),
            "learning_rate": str(LEARNING_RATE),
            "lora_rank": str(LORA_RANK),
        }
    ).result()

    print("Training checkpoint:")
    print(state.path)

    print("\nSaving sampler weights...")

    sampler_checkpoint = training_client.save_weights_for_sampler(
        "trailtutor-sft-v1-sampler"
    ).result()

    print("Sampler checkpoint:")
    print(sampler_checkpoint.path)

    print("\nTinker console:")
    print(training_client.get_console_url())

    try:
        print("\nPlayground:")
        print(sampler_checkpoint.get_playground_url())
    except Exception:
        pass

    print("\nTraining complete.")


if __name__ == "__main__":
    main()