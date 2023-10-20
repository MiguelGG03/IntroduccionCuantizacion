import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import QuantizationConfig, QuantizedGPT2LMHeadModel

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

# Cuantización del input
input_ids = input_ids.to(dtype=torch.int8)

# Realizar la generación de texto
output = quantized_model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

for sequence in output:
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    print("Texto Generado:", generated_text)
