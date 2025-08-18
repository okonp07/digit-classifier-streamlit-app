import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import time
import base64
import os
import torch.nn as nn
import torch.nn.functional as F

# Set page configuration
st.set_page_config(
    page_title="Enhanced Digit Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Model Architecture (same as in your training)
class LightweightDigitCNN(nn.Module):
    def __init__(self, input_channels=13, num_classes=10):
        super(LightweightDigitCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions for the linear layer
        self.fc1 = nn.Linear(64 * 1 * 10, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Add channel dimension: (batch, 13, 87) -> (batch, 1, 13, 87)
        x = x.unsqueeze(1)
        
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Audio Processor
class AudioProcessor:
    def __init__(self, sample_rate=22050, max_duration=1.0, n_mels=13):
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        self.n_mels = n_mels
    
    def load_and_preprocess(self, audio_array, sample_rate):
        """Process audio array and extract features"""
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Pad or trim to fixed length
        audio_array = self.pad_or_trim(audio_array)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_array, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mels,
            n_fft=512,
            hop_length=256
        )
        
        return mfcc
    
    def pad_or_trim(self, audio):
        """Ensure all audio samples have same length"""
        if len(audio) > self.max_length:
            # Trim from center
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        else:
            # Pad with zeros
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio

# Predictor Class
class DigitPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor()
        
        # Initialize model
        self.model = LightweightDigitCNN()
        
        # Load model weights if file exists
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                st.error(f"Error loading model {model_path}: {str(e)}")
                self.model_loaded = False
        else:
            st.warning(f"Model file {model_path} not found. Using random predictions for demo.")
            self.model_loaded = False
    
    def predict_from_array(self, audio_array, sample_rate=22050):
        """Predict digit from numpy audio array"""
        if not self.model_loaded:
            # Return random prediction for demo
            predicted_digit = np.random.randint(0, 10)
            confidence = np.random.uniform(0.7, 0.95)
            probabilities = np.random.dirichlet(np.ones(10))
            probabilities[predicted_digit] = confidence
            probabilities = probabilities / probabilities.sum()
            return predicted_digit, confidence, probabilities
        
        # Extract features
        mfcc = self.processor.load_and_preprocess(audio_array, sample_rate)
        
        # Convert to tensor and add batch dimension
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_digit = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
        
        return predicted_digit, confidence, probabilities[0].cpu().numpy()

# Load models (cached)
@st.cache_resource
def load_models():
    """Load both original and enhanced models"""
    try:
        # Try to load models - adjust paths as needed
        original_predictor = DigitPredictor('lightweight_digit_model.pth')
        enhanced_predictor = DigitPredictor('enhanced_digit_model.pth')
        return original_predictor, enhanced_predictor, True
    except Exception as e:
        st.warning(f"Models not found, using demo mode: {str(e)}")
        # Create demo predictors
        demo_original = DigitPredictor('dummy_original.pth')
        demo_enhanced = DigitPredictor('dummy_enhanced.pth')
        return demo_original, demo_enhanced, False

# Audio processing functions
def process_audio(audio_file, predictor):
    """Process uploaded audio file and make prediction"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process audio
        audio_data, sample_rate = librosa.load(tmp_file_path, sr=22050)
        duration = len(audio_data) / sample_rate
        max_amplitude = np.max(np.abs(audio_data))
        
        # Make prediction
        start_time = time.time()
        predicted_digit, confidence, probabilities = predictor.predict_from_array(audio_data, sample_rate)
        inference_time = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            'success': True,
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities,
            'duration': duration,
            'max_amplitude': max_amplitude,
            'inference_time': inference_time,
            'audio_data': audio_data,
            'sample_rate': sample_rate
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def create_probability_chart(probabilities, predicted_digit):
    """Create interactive probability chart"""
    digits = list(range(10))
    colors = ['#ff7f7f' if i == predicted_digit else '#7f7fff' for i in digits]
    
    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.3f}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Prediction Probabilities (Predicted: {predicted_digit})",
        xaxis_title="Digit",
        yaxis_title="Probability",
        height=400,
        showlegend=False
    )
    
    return fig

def create_waveform_chart(audio_data, sample_rate):
    """Create waveform visualization"""
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Enhanced Spoken Digit Recognition</h1>', unsafe_allow_html=True)
    
    # Load models
    original_predictor, enhanced_predictor, models_loaded = load_models()
    
    if not models_loaded:
        st.info("ℹ️ Running in demo mode - models not found. Upload model files for full functionality.")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Enhanced Model (Recommended)", "Original Model", "Compare Both"]
    )
    
    # Audio analysis options
    show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
    show_probabilities = st.sidebar.checkbox("Show All Probabilities", value=True)
    
    # Information section
    with st.sidebar.expander("ℹ️ About This App"):
        st.write("""
        This app demonstrates an enhanced spoken digit recognition system:
        
        - **Enhanced Model**: Trained on FSDD + Google Speech Commands with data augmentation
        - **Original Model**: Trained only on FSDD dataset
        - **Real-time**: Fast inference (<10ms)
        - **Robust**: Works with real-world audio conditions
        
        **Tips for best results:**
        - Record 1-2 seconds of clear speech
        - Speak digits 0-9 clearly
        - Use quiet environment
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎵 Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload a recording of yourself saying a digit (0-9)"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Audio player
            st.audio(uploaded_file)
            
            # Process button
            if st.button("🚀 Analyze Audio", key="analyze_btn"):
                with st.spinner("Processing audio..."):
                    
                    if model_choice == "Enhanced Model (Recommended)":
                        result = process_audio(uploaded_file, enhanced_predictor)
                        
                        if result['success']:
                            # Display results
                            st.markdown("### 🎯 Enhanced Model Results")
                            
                            col_result1, col_result2, col_result3 = st.columns(3)
                            
                            with col_result1:
                                st.metric("Predicted Digit", result['predicted_digit'])
                            with col_result2:
                                confidence_pct = result['confidence'] * 100
                                st.metric("Confidence", f"{confidence_pct:.1f}%")
                            with col_result3:
                                st.metric("Inference Time", f"{result['inference_time']*1000:.1f}ms")
                            
                            # Confidence assessment
                            if result['confidence'] > 0.9:
                                st.markdown('<div class="success-card">🟢 <strong>Very High Confidence</strong></div>', unsafe_allow_html=True)
                            elif result['confidence'] > 0.7:
                                st.markdown('<div class="metric-card">🟡 <strong>High Confidence</strong></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="warning-card">🟠 <strong>Medium Confidence</strong></div>', unsafe_allow_html=True)
                            
                            # Visualizations
                            if show_probabilities:
                                st.plotly_chart(create_probability_chart(result['probabilities'], result['predicted_digit']), use_container_width=True)
                            
                            if show_waveform:
                                st.plotly_chart(create_waveform_chart(result['audio_data'], result['sample_rate']), use_container_width=True)
                            
                            # Add to history
                            st.session_state.prediction_history.append({
                                'timestamp': time.strftime("%H:%M:%S"),
                                'model': 'Enhanced',
                                'prediction': result['predicted_digit'],
                                'confidence': result['confidence'],
                                'filename': uploaded_file.name
                            })
                            
                        else:
                            st.error(f"❌ Error processing audio: {result['error']}")
                    
                    elif model_choice == "Compare Both":
                        # Process with both models
                        enhanced_result = process_audio(uploaded_file, enhanced_predictor)
                        original_result = process_audio(uploaded_file, original_predictor)
                        
                        if enhanced_result['success'] and original_result['success']:
                            st.markdown("### ⚖️ Model Comparison")
                            
                            # Side-by-side comparison
                            comp_col1, comp_col2 = st.columns(2)
                            
                            with comp_col1:
                                st.markdown("#### 🔴 Enhanced Model")
                                st.metric("Prediction", enhanced_result['predicted_digit'])
                                st.metric("Confidence", f"{enhanced_result['confidence']*100:.1f}%")
                                if show_probabilities:
                                    st.plotly_chart(create_probability_chart(enhanced_result['probabilities'], enhanced_result['predicted_digit']), use_container_width=True, key="enhanced_prob")
                            
                            with comp_col2:
                                st.markdown("#### 🔵 Original Model")
                                st.metric("Prediction", original_result['predicted_digit'])
                                st.metric("Confidence", f"{original_result['confidence']*100:.1f}%")
                                if show_probabilities:
                                    st.plotly_chart(create_probability_chart(original_result['probabilities'], original_result['predicted_digit']), use_container_width=True, key="original_prob")
                            
                            # Comparison summary
                            st.markdown("#### 📋 Comparison Summary")
                            
                            if enhanced_result['predicted_digit'] == original_result['predicted_digit']:
                                st.success(f"✅ Both models agree: Digit **{enhanced_result['predicted_digit']}**")
                            else:
                                st.warning(f"⚠️ Models disagree! Enhanced: **{enhanced_result['predicted_digit']}**, Original: **{original_result['predicted_digit']}**")
    
    with col2:
        st.subheader("📈 Prediction History")
        
        if st.session_state.prediction_history:
            # Display recent predictions
            for pred in reversed(st.session_state.prediction_history[-5:]):
                st.markdown(f"""
                **{pred['timestamp']}** - {pred['model']} Model  
                Predicted: **{pred['prediction']}** ({pred['confidence']*100:.1f}% confidence)
                """)
                st.markdown("---")
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions yet. Upload an audio file to get started!")
        
        # Model performance info
        st.subheader("🏆 Model Performance")
        
        performance_data = {
            "Metric": ["Validation Accuracy", "Real-World Accuracy", "Inference Time", "Model Size"],
            "Enhanced Model": ["94.8%", "90%", "8.5ms", "0.53MB"],
            "Original Model": ["96.6%", "30%", "8.5ms", "0.53MB"]
        }
        
        st.table(performance_data)

if __name__ == "__main__":
    main()
    
    # Footer
    st.markdown("""
    ---
    **Enhanced Spoken Digit Recognition System** | Built with Streamlit 🚀  
    Combining FSDD + Google Speech Commands datasets with advanced data augmentation
    """)
