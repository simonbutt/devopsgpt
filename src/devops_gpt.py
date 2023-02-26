import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import logging
import warnings


class DevopsGPT:
    def __init__(
        self, model_path: str, meta_path: str = "", device: str = "cpu", seed: int = 706
    ) -> None:
        self.device = device
        self._load_ctx(seed)
        self._load_model(model_path)
        self._load_encodings(meta_path)
        logging.info("DevopsGPT loaded")

    def _load_ctx(self, seed: int, dtype: str = "bfloat16") -> None:
        # Enables device contexts for {GPU,CPU} workloads
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = (
            "cuda" if "cuda" in self.device else "cpu"
        )  # for later use in torch.autocast
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

    def _load_model(self, model_path: str):
        # Loads model from ckpt.pt file in model_path
        checkpoint = torch.load(f"{model_path}/ckpt.pt", map_location=self.device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        self.model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def _load_encodings(self, meta_path: str = ""):
        # Loads the encode and decode functions for text ingestion
        if meta_path != "":
            with open(f"{meta_path}/meta.pkl", "rb") as meta_file:
                metadata = pickle.load(meta_file)
            logging.info(f"Loading encodings from {meta_path}")
            stoi, itos = metadata["stoi"], metadata["itos"]
            self.encode = lambda s: [stoi[c] for c in s]
            self.decode = lambda l: "".join([itos[i] for i in l])
        else:
            logging.info("No meta.pkl found, assuming default GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)

    def generate(
        self,
        prompt: str,
        pre_prompt: str = "On google cloud and following devops best practises",
        max_token_len: int = 50,
        temperature: float = 1,
        top_k: int = 200,
    ) -> str:
        full_prompt = f"{pre_prompt} {prompt}"

        ids = self.encode(full_prompt)
        x = torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...]
        logging.debug("ids loaded")
        # Generates response given input prompt
        with torch.no_grad():
            with self.ctx:
                y = self.model.generate(
                    x, max_token_len, temperature=temperature, top_k=top_k
                )
                logging.debug("message generated")
                output = self.decode(y[0].tolist())
                return output.replace(pre_prompt, "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dGPT = DevopsGPT(
        model_path="out_sre_cpu", meta_path="data/sre.google_char", device="cpu"
    )

    print(
        dGPT.generate(
            prompt="The importance of service level objectives are", max_token_len=200
        )
    )
