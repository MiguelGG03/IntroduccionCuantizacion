import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, QuantizationConfig, QuantizedGPT2LMHeadModel
from math import exp

# Cargar el modelo preentrenado GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Cuantización del modelo
quantization_config = QuantizationConfig.from_float(model)
quantized_model = QuantizedGPT2LMHeadModel(quantization_config)
quantized_model.load_state_dict(model.state_dict())

# Generar texto de muestra
prompt = "En un mundo lejano"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Es importante cambiar el modelo para que esté en modo evaluación
quantized_model.eval()
model.eval()

# Cuantización del input
input_ids = input_ids.to(dtype=torch.int8)

# Realizar la generación de texto con ambos modelos
output_original = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
output_quantized = quantized_model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

# Calcular la perplexidad para ambos modelos
def calculate_perplexity(output, model):
    total_log_likelihood = 0
    total_tokens = 0
    for sequence in output:
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
        tokens = tokenizer.encode(generated_text, return_tensors="pt")
        with torch.no_grad():
            logits = model(tokens).logits
        target = tokens.view(-1)
        log_likelihood = logits.view(-1, logits.size(-1)).gather(1, target.unsqueeze(1))
        total_log_likelihood += log_likelihood.sum()
        total_tokens += len(target)
    perplexity = exp(-total_log_likelihood / total_tokens)
    return perplexity

perplexity_original = calculate_perplexity(output_original, model)
perplexity_quantized = calculate_perplexity(output_quantized, quantized_model)

print("Perplexity del modelo original:", perplexity_original)
print("Perplexity del modelo cuantizado:", perplexity_quantized)
