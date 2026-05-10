import streamlit as st
import numpy as np
import pickle
from PIL import Image
import io
import traceback
import time

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------------------- CONFIG ----------------------------
MAX_CAPTION_LEN = 30          
IMG_FEATURE_DIM = 1536        
START_TOKEN = '<start>'   
END_TOKEN   = '<end>'
PAD_TOKEN   = '<pad>'

# PyTorch Architecture Dimensions
EMBED_DIM = 300
ENCODER_DIM = 1536
DECODER_DIM = 1024
ATTENTION_DIM = 256
DEVICE = torch.device('cpu')  # Force CPU for Streamlit / Hugging Face free tier
# ----------------------------------------------------------------

# ---------- Model Architecture Classes ----------

# 1. The Fine-Tuned Encoder
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.extractor = base_model.features
        
        # We don't need freezing/unfreezing logic here because we are only doing inference
        
    def forward(self, images):
        features = self.extractor(images)
        # Reshape is handled directly inside the encoder now!
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        return features

# 2. Attention Layer
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

# 3. LSTM Decoder

class ImageCaptioningModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim, padidx, embedding_tensor=None ):
        super(ImageCaptioningModel, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        if embedding_tensor is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False, padding_idx=padidx)
            print(f"Embedding Trainable Parameters: {self.embedding.weight.requires_grad}") 
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padidx)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell_1 = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.lstm_cell_2 = nn.LSTMCell(decoder_dim, decoder_dim, bias=True)
        self.init_h1 = nn.Linear(encoder_dim, decoder_dim)
        self.init_c1 = nn.Linear(encoder_dim, decoder_dim)
        self.init_h2 = nn.Linear(encoder_dim, decoder_dim)
        self.init_c2 = nn.Linear(encoder_dim, decoder_dim)
        self.dropout_lstm = nn.Dropout(0.5)  
        self.dropout_out  = nn.Dropout(0.5)  
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_length = captions.size(1)
        embeddings = self.embedding(captions)
        mean_encoder_out = features.mean(dim=1)
        h1 = self.init_h1(mean_encoder_out)
        c1 = self.init_c1(mean_encoder_out)
        h2 = self.init_h2(mean_encoder_out)
        c2 = self.init_c2(mean_encoder_out)
        predictions = torch.zeros(batch_size, seq_length, self.vocab_size).to(features.device)

        # ── The Time-Step Loop ──
        for t in range(seq_length):
            context_vector, attention_weights = self.attention(features, h2)
            lstm_input = torch.cat([embeddings[:, t, :], context_vector], dim=1)
            h1, c1 = self.lstm_cell_1(lstm_input, (h1, c1))
            h1_drop = self.dropout_lstm(h1)
            h2, c2 = self.lstm_cell_2(h1_drop, (h2, c2))
            output = self.fc(self.dropout_out(h2))

            predictions[:, t, :] = output

        return predictions

# ---------- Load Models & Weights ----------

@st.cache_resource
def load_feature_extractor(encoder_path):
    """Loads the fine-tuned CNN Encoder."""
    encoder = CNNEncoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE, weights_only=True))
    encoder.to(DEVICE)
    encoder.eval()
    return encoder

@st.cache_resource
def load_captioning_model(model_path, word2idx_path, idx2word_path):
    """Loads the vocabularies and the fine-tuned LSTM Decoder."""
    with open(word2idx_path, 'rb') as f:
        word2idx = pickle.load(f)
    with open(idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)

    vocab_size = len(word2idx)
    
    # Fix
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        encoder_dim=ENCODER_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
        padidx=word2idx['<pad>']
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    return model, word2idx, idx2word

# ---------- Image feature extraction ----------
def extract_features(img, encoder):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Notice we don't reshape here anymore! The CNNEncoder does it.
        feats = encoder(img_tensor)
        
    return feats

# ---------- Caption generation (Beam Search) ----------
def generate_caption_beam(model, image_features, word2idx, idx2word, max_length, beam_width=5):
    """
    Beam Search decoding: Keeps track of top 'k' most probable sequences.
    """
    start_idx = word2idx[START_TOKEN]
    end_idx = word2idx[END_TOKEN]
    pad_idx = word2idx.get(PAD_TOKEN, 0)
    
    # Initial beam: list of tuples (sequence_of_indices, cumulative_log_prob)
    sequences = [([start_idx], 0.0)]

    with torch.no_grad():
        for step in range(max_length):
            all_candidates = []
            
            for seq, score in sequences:
                # If sequence already ended, keep it in the candidates
                if seq[-1] == end_idx:
                    all_candidates.append((seq, score))
                    continue
                
                # Pad the current sequence to MAX_LEN for the model
                padded_seq = seq + [pad_idx] * (max_length - len(seq))
                seq_tensor = torch.tensor([padded_seq], dtype=torch.long).to(DEVICE)
                
                # Forward pass
                outputs = model(image_features, seq_tensor)
                
                # Get prediction for the NEXT word
                next_word_pos = len(seq) - 1
                next_word_logits = outputs[0, next_word_pos, :]
                
                # Convert raw logits to probabilities
                preds = F.softmax(next_word_logits, dim=-1)
                top_probs, top_indices = torch.topk(preds, beam_width)
                top_probs   = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq   = seq + [int(idx)]
                    new_score = score + np.log(prob + 1e-7)
                    all_candidates.append((new_seq, new_score))
                    
            # Sort all candidates by score and keep the best `beam_width` sequences
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # If all sequences in our top beam have ended, stop early
            if all([s[0][-1] == end_idx for s in sequences]):
                break

    # Select the absolute best sequence from our final beams
    best_seq, _ = sequences[0]
    
    # Convert indices to words, removing special tokens
    caption = [idx2word[idx] for idx in best_seq 
             if idx not in (pad_idx, start_idx, end_idx)]
             
    return ' '.join(caption)

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Image Captioning Dashboard", layout="centered")
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image and get an AI‑generated caption.")

    # 4 Files Required Now!
    ENCODER_PATH  = "best_encoder_V7.pth"
    MODEL_PATH    = "best_model_V7.pth"   
    WORD2IDX_PATH = "word2idx_V7.pkl"
    IDX2WORD_PATH = "idx2word_V7.pkl"

    with st.spinner("Loading models... (this may take a moment on first run)"):
        # Pass the encoder path to the extractor
        feature_extractor = load_feature_extractor(ENCODER_PATH)
        caption_model, word2idx, idx2word = load_captioning_model(
            MODEL_PATH, WORD2IDX_PATH, IDX2WORD_PATH
        )
    st.success("Models loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    
    

    if uploaded_file is not None:
        try :
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Add a slider for Beam Width (Defaults to 3, max 10)
            beam_width = st.slider("Beam Width", min_value=1, max_value=10, value=3)

            if st.button("Generate Caption"):
                with st.spinner(f"Generating with Beam Search (Width: {beam_width})..."):
                    # Extract features
                    features = extract_features(image, feature_extractor)
                    
                    # Generate caption using the NEW Beam Search function
                    caption = generate_caption_beam(
                        caption_model, features, word2idx, idx2word, MAX_CAPTION_LEN, beam_width=beam_width
                    )
                    
                # --- TYPEWRITER EFFECT ---
                # 1. Create an empty container on the screen
                placeholder = st.empty()
                displayed_text = "### 📝 Caption: "
                
                # 2. Split the generated caption into individual words
                words = caption.capitalize().split()
                
                # 3. Loop through the words and update the container
                for word in words:
                    displayed_text += word + " "
                    # The '▌' character adds a cool blinking terminal cursor effect!
                    placeholder.markdown(displayed_text + "▌") 
                    time.sleep(0.15)  # Delay 0.15 seconds between words
                
                # 4. Print the final version to remove the cursor block
                placeholder.markdown(displayed_text + "")

        except Exception as e:
            st.error(f"Upload error:\n\n```\n{e}\n```")
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
