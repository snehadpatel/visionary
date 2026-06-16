"""
Visionary VLM - TinyLlama-1.1B Language Decoder.
Generates structured text responses conditioned on visual embeddings.
Runs on Apple MPS with float32 precision.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import torch



def stream_generate_response(
    prompt_text: str,
    image_embedding: torch.Tensor,
    max_new_tokens: int = 512,
):
    """
    Generator that yields tokens one by one for real-time streaming.
    Uses TextIteratorStreamer in a separate thread.
    """
    _load_model()

    system = (
        "You are Visionary, an expert AI interior designer. "
        "You analyse room photos and give structured design recommendations."
    )

    SYS_TAG = "<" + "|system|" + ">"
    USR_TAG = "<" + "|user|" + ">"
    AST_TAG = "<" + "|assistant|" + ">"

    chat_prompt = (
        SYS_TAG + "\n" + system + "\n"
        + USR_TAG + "\n[ROOM IMAGE EMBEDDED]\n" + prompt_text + "\n"
        + AST_TAG + "\n"
    )

    inputs = _tokenizer(chat_prompt, return_tensors="pt").to(device)
    
    # Setup streamer
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        token_embeds = _llm.get_input_embeddings()(inputs["input_ids"])
        
        if image_embedding.shape[-1] != 2048:
            padding = torch.zeros(1, 2048 - image_embedding.shape[-1], device=device, dtype=_dtype)
            visual_prefix = torch.cat([image_embedding.to(device).to(_dtype), padding], dim=1).unsqueeze(1)
        else:
            visual_prefix = image_embedding.unsqueeze(1).to(device).to(_dtype)

        combined = torch.cat([visual_prefix, token_embeds], dim=1)
        
        # Expand attention mask for the visual token
        attention_mask = inputs["attention_mask"]
        visual_mask = torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        generation_kwargs = dict(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
        
        # Start generation in a separate thread
        thread = Thread(target=_llm.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

# Device selection
# Using MPS + Float16 for extreme speed and low memory
device = "mps"
_dtype = torch.float16

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_tokenizer = None
_llm = None


def _load_model():
    global _tokenizer, _llm
    if _tokenizer is not None:
        return

    print(f"[VLM Decoder] Loading TinyLlama-1.1B on {device} ({_dtype})...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype,
        device_map={"": device},
    )
    _llm.eval()
    print("[VLM Decoder] TinyLlama-1.1B ready.")


def generate_response(
    prompt_text: str,
    image_embedding: torch.Tensor,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate text conditioned on both the prompt and the visual embedding.
    The image embedding is injected as a virtual visual prefix token before
    the text tokens, allowing the LLM to attend to visual information.
    """
    _load_model()

    system = (
        "You are Visionary, an expert AI interior designer. "
        "You analyse room photos and give structured design recommendations. "
        "Always respond in valid JSON when asked for structured output."
    )

    SYS_TAG = "<" + "|system|" + ">"
    USR_TAG = "<" + "|user|" + ">"
    AST_TAG = "<" + "|assistant|" + ">"

    chat_prompt = (
        SYS_TAG + "\n" + system + "\n"
        + USR_TAG + "\n[ROOM IMAGE EMBEDDED]\n" + prompt_text + "\n"
        + AST_TAG + "\n"
    )

    inputs = _tokenizer(chat_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        # Inject visual embedding by adding it to the first token embeddings
        token_embeds = _llm.get_input_embeddings()(inputs["input_ids"])
        
        # Project 1280 (SceneNet) to 2048 (TinyLlama)
        if image_embedding.shape[-1] != 2048:
            padding = torch.zeros(1, 2048 - image_embedding.shape[-1], device=device, dtype=_dtype)
            visual_prefix = torch.cat([image_embedding.to(device).to(_dtype), padding], dim=1).unsqueeze(1)
        else:
            visual_prefix = image_embedding.unsqueeze(1).to(device).to(_dtype)
            
        combined = torch.cat([visual_prefix, token_embeds], dim=1)
        
        # Expand attention mask for the visual token
        attention_mask = inputs["attention_mask"]
        visual_mask = torch.ones((1, 1), device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        output = _llm.generate(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    decoded = _tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the assistant response part
    if AST_TAG in decoded:
        decoded = decoded.split(AST_TAG)[-1].strip()
    return decoded
