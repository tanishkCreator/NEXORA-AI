
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# NEXORA AI MODEL
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

print("Loading NEXORA AI...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model.eval()

print("NEXORA AI is ready!")


def nexora_chat(message, history):
    if not message.strip():
        return "Please enter a programming question."

    messages = [
        {
            "role": "system",
            "content": (
                "You are NEXORA AI, a coding assistant. "
                "You specialize in Python, C, C++ and Java. "
                "Explain code clearly and provide examples."
            )
        },
        {
            "role": "user",
            "content": message
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer


demo = gr.ChatInterface(
    fn=nexora_chat,
    title="NEXORA AI",
    description="Your coding assistant for Python, C, C++ and Java."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
