import os
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# 頁面配置
st.set_page_config(
    page_title="紅尾蚺基因 AI 辨識系統", 
    page_icon="🐍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 精美 UI 樣式注入 (CSS 美化)
st.markdown("""
<style>
    /* 主體背景色調 */
    .main {
        background-color: #f4f6f9;
    }
    /* 標題樣式 */
    h1, h2, h3 {
        color: #1b5e20;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* 分頁籤美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7d32 !important;
        color: white !important;
    }
    /* 按鈕美化 */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        background-color: #2e7d32;
        color: white;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(46, 125, 50, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        box-shadow: 0 6px 8px rgba(27, 94, 32, 0.3);
    }
    /* 資訊框卡片化 */
    div.stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 1. 基因資料庫設定 (MorphMarket 標準 + Het 選項)
GENES_CODOM = [
    "Hypo (Salmon)", "Super Hypo", "Jungle", "Motley", "Stripe", "Fire", "Super Fire", 
    "IMG (Increased Melanin Gene)", "Super IMG", "Arabesque", "Inca", "Aztec", 
    "Keltic", "Key West", "Labyrinth", "Roswell", "Onyx", "Black Pastel", "Tiger", "Jaguar", "Harlequin"
]

GENES_RECESSIVE = [
    "Albino (Kahl)", "Albino (Sharp)", "Albino (VPI)", "Anerythristic (Type 1)", 
    "Anerythristic (Type 2 / Nicaraguan)", "Blood", "Caramel Albino (T+)", "Sterling Albino (T+)", "Leopard", "Piebald"
]

# 自動將隱性基因擴充為對應的 Het 選項
GENES_HET = [f"Het {g}" for g in GENES_RECESSIVE]

GENES_POLYGENIC = [
    "Pastel", "Patternless", "Ladderback", "High Contrast", "Red Strain", "Paradox", "Mandarin Belly"
]

ALL_GENES = GENES_CODOM + GENES_RECESSIVE + GENES_HET + GENES_POLYGENIC
DB_FILE = "boa_dataset.json"

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==========================================
# 側邊欄：資料庫備份與還原管理
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/snake.png", width=70)
    st.title("控制與備份中心")
    st.divider()
    
    current_db = load_db()
    st.metric(label="📊 資料庫總訓練個體數", value=f"{len(current_db)} 筆")
    
    st.markdown("### 💾 雲端資料安全")
    if len(current_db) > 0:
        json_str = json.dumps(current_db, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 下載備份資料庫 (JSON)",
            data=json_str,
            file_name="boa_dataset_backup.json",
            mime="application/json"
        )
    
    st.markdown("---")
    st.markdown("### ♻️ 還原資料庫")
    uploaded_backup = st.file_uploader("上傳備份檔案還原", type=["json"], key="backup_upload")
    if uploaded_backup is not None:
        try:
            imported_data = json.load(uploaded_backup)
            if isinstance(imported_data, list):
                save_db(imported_data)
                st.success(f"成功還原！已載入 {len(imported_data)} 筆資料。")
                st.rerun()
            else:
                st.error("格式錯誤。")
        except Exception as e:
            st.error(f"還原失敗: {e}")

# 2. 視覺特徵提取引擎
def extract_visual_features(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None:
        return None
    img_resized = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
    s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
    v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (224 * 224)
    contrast = np.std(gray)
    return [h_mean, h_std, s_mean, s_std, v_mean, v_std, edge_density, contrast]

def process_multiple_photos(image_files):
    all_feats = []
    for img_file in image_files:
        feats = extract_visual_features(img_file)
        if feats is not None:
            all_feats.append(feats)
    if not all_feats:
        return None
    return np.mean(all_feats, axis=0).tolist()

# 3. AI 模型訓練與預測
def train_model():
    db = load_db()
    if len(db) < 2:
        return None
    X = [item["features"] for item in db]
    Y = [[1 if gene in item["genes"] else 0 for gene in ALL_GENES] for item in db]
    base_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    model = MultiOutputClassifier(base_rf)
    model.fit(X, Y)
    return model

# 4. 主介面設計
st.title("🐍 紅尾蚺 (Boa) 智慧基因辨識系統")
st.markdown("透過多圖特徵分析與 AI機器學習，精準辨識紅尾蚺的顯性、隱性、Het 與超級體組合。")
st.divider()

tab1, tab2 = st.tabs(["🧬 1. 訓練資料庫錄入 (Input)", "🔍 2. 待測個體基因判讀 (Inference)"])

with tab1:
    st.subheader("📥 錄入新個體訓練資料")
    uploaded_files = st.file_uploader("上傳該個體的照片（可多張：頭部、背部、側腹等）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    st.markdown("### 🧬 勾選該個體擁有的基因標籤：")
    
    # 採用優雅的欄位分類排版
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("⭐ **共顯性基因 & 超級體 (Co-Dom)**")
            selected_codom = [g for g in GENES_CODOM if st.checkbox(g, key=f"c_{g}")]
            
            st.markdown("🧬 **隱性基因 (Recessive - 顯性表現)**")
            selected_rec = [g for g in GENES_RECESSIVE if st.checkbox(g, key=f"r_{g}")]
            
    with col2:
        with st.container():
            st.markdown("🧪 **Het 基因 (Heterozygous 攜帶者)**")
            selected_het = [g for g in GENES_HET if st.checkbox(g, key=f"h_{g}")]
            
            st.markdown("🎨 **選育表現型 (Polygenic)**")
            selected_poly = [g for g in GENES_POLYGENIC if st.checkbox(g, key=f"p_{g}")]
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 寫入資料庫並訓練 AI", use_container_width=True):
        if not uploaded_files:
            st.error("⚠️ 請至少上傳一張照片！")
        else:
            all_selected = selected_codom + selected_rec + selected_het + selected_poly
            feats = process_multiple_photos(uploaded_files)
            if feats is not None:
                db = load_db()
                db.append({
                    "id": len(db) + 1, 
                    "genes": all_selected, 
                    "features": feats, 
                    "photo_count": len(uploaded_files)
                })
                save_db(db)
                st.success(f"🎉 成功寫入資料庫！目前總樣本數：{len(db)} 筆")
            else:
                st.error("⚠️ 圖片解析失敗，請重新上傳。")

with tab2:
    st.subheader("🔍 上傳待辨識紅尾蚺照片")
    predict_files = st.file_uploader("上傳要檢測的個體照片（建議上傳多張以提高準確率）", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="pred")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 開始 AI 基因分析", use_container_width=True):
        if not predict_files:
            st.error("⚠️ 請至少上傳一張待測照片！")
        else:
            model = train_model()
            if model is None:
                st.warning("⚠️ 資料庫樣本數不足（至少需要 2 筆以上個體資料才能開始預測）。")
            else:
                feats = process_multiple_photos(predict_files)
                if feats is not None:
                    probabilities = model.predict_proba(np.array([feats]))
                    results = {}
                    predicted_combos = []
                    for idx, gene_name in enumerate(ALL_GENES):
                        prob = probabilities[idx][0][1] if len(probabilities[idx][0]) > 1 else 0.0
                        pct = round(prob * 100, 1)
                        results[gene_name] = pct
                        if pct >= 50.0:
                            predicted_combos.append(f"{gene_name} ({pct}%)")
                    
                    st.markdown("### 📊 AI 綜合基因預測結果：")
                    if predicted_combos:
                        st.info(" ＋ ".join(predicted_combos))
                    else:
                        st.success("表現型判斷：普通野生型 / Normal")
                        
                    st.markdown("### 📈 各基因獨立命中信心度：")
                    # 以卡片或進度條呈現信心度
                    for g, p in sorted(results.items(), key=lambda x: x[1], reverse=True):
                        if p > 10:  # 只顯示信心度大於 10% 的項目讓版面更清爽
                            col_a, col_b = st.columns([3, 7])
                            with col_a:
                                st.write(f"**{g}**")
                            with col_b:
                                st.progress(p / 100.0, text=f"{p}%")
