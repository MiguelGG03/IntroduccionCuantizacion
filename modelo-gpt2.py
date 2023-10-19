from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo preentrenado GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#Limpiar terminal
print("\033c")

# Generar texto de muestra
prompt = "En un mundo lejano"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

for sequence in output:
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    print("Texto Generado:", generated_text)
