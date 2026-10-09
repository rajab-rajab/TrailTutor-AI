from tinker import ServiceClient

client = ServiceClient()
caps = client.get_server_capabilities()

for model in caps.supported_models:
    if model.trainable:
        print(
            f"{model.model_name} | "
            f"trainable={model.trainable} | "
            f"sampleable={model.sampleable} | "
            f"context={model.max_context_length}"
        )
