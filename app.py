import streamlit as st
import numpy as np
import pickle
from PIL import Image
import io
import traceback

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------------------- CONFIG ----------------------------
MAX_CAPTION_LEN = 30          # Updated to your PyTorch training length
IMG_FEATURE_DIM = 1536        # EfficientNetB3 feature size
START_TOKEN = '<start>'   
END_TOKEN   = '<end>'
PAD_TOKEN   = '<pad>'

# PyTorch Architecture Dimensions
EMBED_DIM = 300
ENCODER_DIM = 1536
DECODER_DIM = 512
ATTENTION_DIM = 256
DEVICE = torch.device('cpu')  # Force CPU for Streamlit / Hugging Face free tier
# ----------------------------------------------------------------

# ---------- Model Architecture Classes ----------
# PyTorch requires these to be defined in the script to load the weights
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(encoder_dim, attention_dim)
        self.W2 = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        hidden_expanded = hidden_state.unsqueeze(1)
        attention_hidden = torch.tanh(self.W1(features) + self.W2(hidden_expanded))
        score = self.V(attention_hidden)
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim):
        super(ImageCaptioningModel, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        
        # Stacked LSTM Cells (Matches your trained .pth file)
        self.lstm_cell_1 = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.lstm_cell_2 = nn.LSTMCell(decoder_dim, decoder_dim, bias=True)
        
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_length = captions.size(1)
        embeddings = self.embedding(captions)
        
        # Initialize hidden states for BOTH LSTMs
        h1 = torch.zeros((batch_size, self.decoder_dim)).to(features.device)
        c1 = torch.zeros((batch_size, self.decoder_dim)).to(features.device)
        h2 = torch.zeros((batch_size, self.decoder_dim)).to(features.device)
        c2 = torch.zeros((batch_size, self.decoder_dim)).to(features.device)
        
        predictions = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)
        
        for t in range(seq_length):
            # Attention uses the hidden state of the SECOND LSTM
            context_vector, attention_weights = self.attention(features, h2)
            
            # Input for the first LSTM
            lstm_input = torch.cat([embeddings[:, t, :], context_vector], dim=1)
            
            # Pass through First LSTM
            h1, c1 = self.lstm_cell_1(lstm_input, (h1, c1))
            
            # Apply Dropout between LSTMs
            h1_drop = self.dropout(h1)
            
            # Pass through Second LSTM
            h2, c2 = self.lstm_cell_2(h1_drop, (h2, c2))
            
            # Final prediction using the second LSTM's output
            output = self.fc(self.dropout(h2))
            predictions[:, t, :] = output
            
        return predictions

# ---------- Load pre-trained CNN for feature extraction ----------
@st.cache_resource
def load_feature_extractor():
    base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    extractor = base_model.features
    extractor.to(DEVICE)
    extractor.eval()
    return extractor

# ---------- Load captioning model and vocabularies ----------
@st.cache_resource
def load_captioning_model(model_path, word2idx_path, idx2word_path):
    with open(word2idx_path, 'rb') as f:
        word2idx = pickle.load(f)
    with open(idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)

    vocab_size = len(word2idx)
    
    # Instantiate the PyTorch model
    model = ImageCaptioningModel(vocab_size, EMBED_DIM, ENCODER_DIM, DECODER_DIM, ATTENTION_DIM)
    
    # Load the weights (map_location=DEVICE ensures it loads safely on CPU)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    return model, word2idx, idx2word

# ---------- Image feature extraction ----------
def extract_features(img, model):
    # Matches PyTorch training pipeline
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ensure image is RGB (drops alpha channel automatically)
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        feats = model(img_tensor)
        # Reshape to match (1, 100, 1536) sequence format
        feats = feats.view(feats.size(0), feats.size(1), -1).permute(0, 2, 1)
        
    return feats

# ---------- Caption generation (greedy search) ----------
def generate_caption(model, image_features, word2idx, idx2word, max_length):
    """
    Greedy decoding: Translated directly from your TF code.
    Starts with '<start>' and predicts the next word step-by-step.
    """
    start_idx = word2idx[START_TOKEN]
    end_idx = word2idx[END_TOKEN]
    pad_idx = word2idx.get(PAD_TOKEN, 0)
    
    input_seq = [start_idx]
    caption = []

    with torch.no_grad():
        for i in range(max_length):
            # Pad the current sequence to max_length
            padded = input_seq + [pad_idx] * (max_length - len(input_seq))
            seq_tensor = torch.tensor([padded], dtype=torch.long).to(DEVICE)
            
            # Predict next word distribution
            outputs = model(image_features, seq_tensor)
            
            # Get the prediction for the NEXT word
            next_word_pos = len(input_seq) - 1
            next_word_logits = outputs[0, next_word_pos, :]
            
            # Get index of highest probability (Greedy Search)
            next_idx = torch.argmax(next_word_logits).item()
            
            if next_idx == end_idx:
                break
            if next_idx != start_idx and next_idx != pad_idx:
                caption.append(idx2word[next_idx])
                
            input_seq.append(next_idx)

    return ' '.join(caption)

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Image Captioning Dashboard", layout="centered")
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image and get an AI‑generated caption.")

    # Paths to your PyTorch artifacts
    MODEL_PATH = "best_model.pth"   # Changed from .keras to .pth
    WORD2IDX_PATH = "word2idx.pkl"
    IDX2WORD_PATH = "idx2word.pkl"

    # Load everything once
    with st.spinner("Loading models... (this may take a moment on first run)"):
        feature_extractor = load_feature_extractor()
        caption_model, word2idx, idx2word = load_captioning_model(
            MODEL_PATH, WORD2IDX_PATH, IDX2WORD_PATH
        )
    st.success("Models loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:

        try :
            # Display the image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Generate Caption"):
                with st.spinner("Generating..."):
                    # Extract features
                    features = extract_features(image, feature_extractor)
                    # Generate caption
                    caption = generate_caption(
                        caption_model, features, word2idx, idx2word, MAX_CAPTION_LEN
                    )
                st.markdown(f"### 📝 Caption: **{caption.capitalize()}**")

        except Exception as e:
            st.error(f"Upload error:\n\n```\n{e}\n```")
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()