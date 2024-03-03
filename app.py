from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Initialize Flask app
app = Flask(__name__)

# Load custom-trained sentence transformer model
model_name = "model.pth"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to encode query and calculate cosine similarity
def calculate_similarity(query):
    # Tokenize input query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the output embeddings (CLS token embeddings)
    embeddings_query = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings
    
    # Calculate cosine similarity with reference embeddings
    reference_embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]) 
    cos_sim = cosine_similarity(embeddings_query, reference_embeddings)
    
    return cos_sim

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        cos_sim = calculate_similarity(query)
        # You can process the cosine similarity scores and display search results here
        return render_template('result.html', query=query, cos_sim=cos_sim)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
