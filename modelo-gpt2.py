from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # Nombre del modelo preentrenado (puedes cambiarlo según tus necesidades)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

input_text = input("Ingrese el texto:")
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
