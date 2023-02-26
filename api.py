from fastapi import FastAPI
from src.devops_gpt import DevopsGPT


app = FastAPI(
    title="DevOps GPT",
    description="GPT-2 large (774M) model, fine tuned on SRE Google books",
    version="0.1.0",
)

dGPT = DevopsGPT(
    model_path="out_sre_long_train", meta_path="data/sre.google_char", device="cuda"
)


@app.get("/api/healthz")
async def health_check():
    return {"api_healthy": True}


@app.post("/api/generate")
async def generate(
    prompt: str,
    pre_prompt: str = "On google cloud and following devops best practises",
    max_token_len: int = 100,
    iterations: int = 1,
) -> str:
    return dGPT.generate(prompt=prompt, pre_prompt="", max_token_len=max_token_len)


if __name__ == "__main__":
    import uvicorn
    import logging

    logging.basicConfig(level=logging.DEBUG)

    uvicorn.run(app, host="0.0.0.0", port=8000)
