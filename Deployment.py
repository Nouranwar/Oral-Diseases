import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oral Disease Detector",
    page_icon="🦷",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    .header-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }

    .header-box h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .header-box p {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }

    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
    }

    .prediction-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e3a5f;
    }

    .confidence-text {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    .disclaimer {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 1.5rem;
    }

    .stProgress > div > div {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis','Hypodontia' , 'Tooth Discoloration', 'Ulcers']
MODEL_PATH = r"D:\CV Projects\Oral Diseases\best_model.pth"

CLASS_INFO = {
    'Calculus': {
        'desc': 'تراكم الجير على الأسنان. يُنصح بزيارة طبيب الأسنان لإزالته بشكل احترافي.',
        'color': '#64748b'
    },
    'Caries': {
        'desc': 'تسوس الأسنان. يحتاج علاجاً مبكراً لمنع التفاقم.',
        'color': '#dc2626'
    },
    'Gingivitis': {
        'desc': 'التهاب اللثة. الصحة الفموية الجيدة تساعد في العلاج.',
        'color': '#b91c1c'
    },
    'Ulcers': {
        'desc': 'قرح الفم. قد تكون مؤلمة وتحتاج إلى رعاية طبية.',
        'color': '#7c3aed'
    },
    'Tooth Discoloration': {
        'desc': 'تلون الأسنان. يمكن علاجه بالتبييض أو القشرة.',
        'color': '#d97706'
    },
    'Hypodontia': {
        'desc': 'غياب أسنان خلقي. يحتاج تدخلاً تقويمياً أو زراعياً.',
        'color': '#0369a1'
    },
}

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # إزالة الـ prefix "model." من كل key
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# ─── Transform ─────────────────────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ─── Predict ───────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return probs.numpy()

# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🦷 Oral Disease Detector</h1>
    <p>ارفع صورة للفم وسيتعرف النموذج على المرض المحتمل باستخدام الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("جاري تحميل النموذج..."):
    try:
        model = load_model()
        st.success("✅ تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"❌ فشل تحميل النموذج: {e}")
        st.stop()

# Upload
uploaded_file = st.file_uploader(
    "اختر صورة للفم أو الأسنان",
    type=["jpg", "jpeg", "png",".jfif"],
    help="يُفضل أن تكون الصورة واضحة وتُظهر منطقة الفم بشكل جيد"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)

    with col2:
        with st.spinner("جاري التحليل..."):
            probs = predict(image, model)

        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx]) * 100
        info = CLASS_INFO[top_class]

        st.markdown(f"""
        <div class="result-card">
            <div class="prediction-label" style="color:{info['color']};">{top_class}</div>
            <div class="confidence-text">نسبة الثقة: <strong>{top_conf:.1f}%</strong></div>
            <hr style="margin:1rem 0; border-color:#e2e8f0;">
            <p style="font-size:0.9rem; color:#475569;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    # All probabilities
    st.markdown("### 📊 نتائج جميع الأمراض")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        label = CLASS_NAMES[i]
        conf = float(probs[i])
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.markdown(f"**{label}**")
            st.progress(conf)
        with col_r:
            st.markdown(f"<br><strong>{conf*100:.1f}%</strong>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>تنبيه:</strong> هذا النظام مساعد تشخيصي فقط ولا يُغني عن استشارة طبيب الأسنان المختص.
        النتائج تعتمد على الصورة المُدخلة وقد لا تكون دقيقة في جميع الحالات.
    </div>
    """, unsafe_allow_html=True)