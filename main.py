from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    pdf_file = request.files['pdf_file']
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    input_tokenized = tokenizer.encode(text, return_tensors='pt',max_length=1024,truncation=True)
    summary_ids = model.generate(input_tokenized,
                                      num_beams=9,
                                      no_repeat_ngram_size=3,
                                      length_penalty=2.0,
                                      min_length=150,
                                      max_length=250,
                                      early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
    return {'summary': summary}

if __name__ == '__main__':
    app.run(debug=True)
