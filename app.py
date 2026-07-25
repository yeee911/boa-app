import os
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

st.set_page_config(page_title="紅尾蚺基因 AI 辨識系統", page_icon="🐍", layout="wide")

# 1. 基因資料庫設定 (MorphMarket 標準)
GENES_CODOM = [
    "Hypo (Salmon)", "Super Hypo", "Jungle", "Motley", "Stripe", "Fire", "Super Fire", 
    "IMG (Increased Melanin Gene)", "Super IMG", "Arabesque", "Inca", "Aztec", 
    "Keltic", "Key West", "Labyrinth", "Roswell", "Onyx", "Black Pastel", "Tiger", "Jaguar", "Harlequin"
]
GENES_RECESSIVE = [
    "Albino (Kahl)", "Albino (Sharp)", "Albino (VPI)", "Anerythristic (Type 1)", 
    "Anerythristic (Type 2 / Nicaraguan)", "Blood", "Caramel Albino (T+)", "Sterling Albino (T+)", "Leopard", "Piebald"
]
GENES_POLYGENIC = [
    "Pastel", "Patternless", "Ladderback", "High Contrast", "Red Strain", "Paradox", "Mandarin Belly"
]

ALL_GENES = GENES_CODOM + GENES_RECESSIVE + GENES_POLYGENIC
DB_FILE = "boa_dataset.json"

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def load_db():
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 2. 視覺特徵提取
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

# 4. 網頁介面設計 (Streamlit)
st.title("🐍 紅尾蚺 (Boa) 基因 AI 學習與辨識系統")

tab1, tab2 = st.tabs(["1. 訓練資料庫錄入 (Input Data)", "2. 待測紅尾蚺基因判讀 (Inference)"])

with tab1:
    st.subheader("上傳新個體照片與基因標籤")
    uploaded_files = st.file_uploader("選擇多張照片（頭部、背部、側腹等）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    st.markdown("#### 勾選該個體擁有的基因：")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**共顯性 / 顯性基因**")
        selected_codom = [g for g in GENES_CODOM if st.checkbox(g, key=f"c_{g}")]
    with col2:
        st.write("**隱性基因**")
        selected_rec = [g for g in GENES_RECESSIVE if st.checkbox(g, key=f"r_{g}")]
    with col3:
        st.write("**選育表現型**")
        selected_poly = [g for g in GENES_POLYGENIC if st.checkbox(g, key=f"p_{g}")]
        
    if st.button("寫入資料庫並訓練 AI", type="primary"):
        if not uploaded_files:
            st.error("請至少上傳一張照片！")
        else:
            all_selected = selected_codom + selected_rec + selected_poly
            feats = process_multiple_photos(uploaded_files)
            if feats is not None:
                db = load_db()
                db.append({"id": len(db)+1, "genes": all_selected, "features": feats, "photo_count": len(uploaded_files)})
                save_db(db)
                st.success(f"成功寫入！當前資料庫總筆數：{len(db)} 筆")
            else:
                st.error("圖片解析失敗，請重新上傳。")

with tab2:
    st.subheader("上傳待辨識蛇的照片")
    predict_files = st.file_uploader("上傳待測照片（可多張）", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="pred")
    
    if st.button("開始 AI 基因分析", type="primary"):
        if not predict_files:
            st.error("請上傳至少一張待測照片！")
        else:
            model = train_model()
            if model is None:
                st.warning("資料庫樣本數不足（至少需要 2 筆以上個體資料才能開始預測）。")
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
                    
                    st.markdown("### 📊 預測結果綜合評估：")
                    if predicted_combos:
                        st.info(" + ".join(predicted_combos))
                    else:
                        st.write("表現型判斷：普通野生型 / Normal")
                        
                    st.markdown("### 各基因獨立信心度：")
                    for g, p in sorted(results.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- **{g}**: `{p}%`")
