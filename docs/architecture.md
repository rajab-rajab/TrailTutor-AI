# Architecture

```text
Browser
  |
  v
TrailTutor FastAPI application
  |
  +--> Gemma baseline endpoint
  |
  +--> Tuned model endpoint
  |
  +--> Evaluation comparison
```

## Deployment principle

DigitalOcean hosts the public application and can also host model-serving infrastructure if your chosen setup supports it.

The application uses environment variables instead of embedding credentials.

## Separation of concerns

Training and serving are intentionally separate.

- Tinker is used for specialization/fine-tuning.
- The trained checkpoint can be served through a compatible model endpoint.
- Gemma remains an independent open-weight baseline path.
- The application can compare both.

This architecture makes the hackathon evidence easier to explain and reproduce.
