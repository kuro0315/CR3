from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    # 4ビット量子化の設定を作成
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    MODEL_ID = "Aratako/c4ai-command-r-v01-japanese-instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # 4ビット量子化して読み込む
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                quantization_config=quantization_config)
    
    input_text = "Pythonで画像を読み込んで表示するプログラムを書いてみよう。"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.7)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)

if __name__ == "__main__":
	main()
