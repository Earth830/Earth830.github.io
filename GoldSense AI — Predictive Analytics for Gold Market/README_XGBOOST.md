# XGBoost Time-Series/Tabular Modeling — Detailed Portfolio README

[![Python](https://img.shields.io/badge/Python-3.x-blue)](#) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](#) [![XGBoost](https://img.shields.io/badge/XGBoost-1.x-brightgreen)](#)

โปรเจกต์นี้สร้างโมเดล **XGBRegressor** จาก `xgboost.ipynb` เพื่อพยากรณ์/รีเกรสชันบนข้อมูลที่ดึงจาก **GC=F (Gold Futures) via yfinance** พร้อมอธิบายโค้ด เวิร์กโฟลว์ เมตริกผลลัพธ์ และกราฟทีละรูป
เหมาะสำหรับแสดงใน **GitHub Portfolio** และแนบใน **เรซูเม่สมัครงาน** สาย Data/ML.

---

## 1) ข้อมูลและการเตรียมข้อมูล (Data & Preprocessing)
- **แหล่งข้อมูล**: GC=F (Gold Futures) via yfinance
- **คอลัมน์เป้าหมาย (y)**: ราคาปลายทาง/ตัวแปรเป้าหมาย (ดูในโน้ตบุ๊กสำหรับคอลัมน์จริง)
- **Features (X)**: ฟีเจอร์ที่สร้างจากข้อมูลดิบ (เช่น lag features, rolling mean/volatility) — *หากมี*
- **การแบ่งข้อมูล**: `train_test_split(test_size=..., random_state=42)` (ดูขนาดสัดส่วนในโน้ตบุ๊ก)
- **Scaling/Encoding**: ไม่จำเป็นต่อ XGBoost เสมอไป (tree-based) แต่ควรทำความสะอาด/จัดรูปแบบให้เรียบร้อย

> เคล็ดลับ: สำหรับ time-series ให้ใช้วิธี split แบบ time-based (ไม่สลับลำดับเวลา) และทำ **walk-forward validation** หากต้องการความเคร่งครัด

---

## 2) โมเดลและพารามิเตอร์ (Model & Hyperparameters)
โมเดล: **XGBRegressor**  

พารามิเตอร์หลักที่ใช้ (จากโน้ตบุ๊ก):
- `n_estimators`: **500**
- `max_depth`: **5**
- `learning_rate`: **0.05**
- `subsample`: **0.7**
- `colsample_bytree`: **0.7**
- `random_state`: **42**

**เหตุผลการตั้งค่า (แนวทางทั่วไป):**
- `learning_rate` ต่ำ (0.05) + `n_estimators` สูง ช่วยให้เรียนรู้ละเอียดขึ้น และลด overfit เมื่อใช้ early stopping
- `subsample` และ `colsample_bytree` < 1.0 ลดความสัมพันธ์ระหว่างต้นไม้ ช่วยลด variance
- `max_depth=5` สร้างพจน์ปฏิสัมพันธ์ได้พอสมควรโดยไม่ลึกเกินไป

> แนะนำให้ **tune เพิ่มเติม** เช่น Grid/Random/Bayes search บน `(max_depth, learning_rate, n_estimators, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)`

---

## 3) เวิร์กโฟลว์ของโค้ด (Code Walkthrough)

1. **Load Data** — ดึง GC=F จาก `yfinance` แล้วเลือกคอลัมน์ที่ต้องการเป็น `y`
2. **Feature Engineering (optional)** — สร้าง lag/rolling features หากต้องการเพิ่มสัญญาณให้โมเดล
3. **Split** — แยก train/test (time-based) และกำหนด `random_state=42` เพื่อทำซ้ำได้
4. **Train** — สร้าง `XGBRegressor` ด้วยพารามิเตอร์ด้านบน และ `fit(X_train, y_train)`  
5. **Evaluate** — คำนวณ **MAE, RMSE** (และสามารถเสริม R², MAPE เพิ่มได้)
6. **Visualize** — พล็อตเส้น **Actual vs Predicted** (และอาจเพิ่ม Feature Importance, Residuals, SHAP)

โค้ดตัวอย่างการสร้างโมเดลสั้น ๆ:
```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 4) ผลลัพธ์และการตีความ (Results & Interpretation)

### ตารางเมตริก
| Metric | Value |
|---|---:|
| **MAE** | **19.32** |
| **RMSE** | **26.45** |

**อ่านค่าอย่างไร?**  
- **MAE ≈ 19.3**: คลาดเคลื่อนเฉลี่ยในหน่วย “ราคา” — เหมาะกับการสื่อสารเชิงธุรกิจ (เข้าใจง่าย)  
- **RMSE ≈ 26.4**: ลงโทษ error ขนาดใหญ่หนักกว่า (เพราะยกกำลังสอง) — ดีสำหรับจับเหตุการณ์พลาดหนัก

> หากต้องการ **R²** และ **MAPE** เพิ่มเติม ให้เพิ่มโค้ดคำนวณดังนี้:
```python
from sklearn.metrics import r2_score, mean_absolute_percentage_error
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"R2: {r2:.3f}, MAPE: {mape:.2f}%")
```

### กราฟผลลัพธ์ (แสดงจากเอาต์พุตที่มีในโน้ตบุ๊ก)

**รูป: Actual vs Predicted (Overview)**  
ดาวน์โหลด: [xgb_fig_01_XGBoost-Forecast-vs-Actual.png](sandbox:/mnt/data/xgb_media/xgb_fig_01_XGBoost-Forecast-vs-Actual.png)  

![](xgb_media/xgb_fig_01_XGBoost-Forecast-vs-Actual.png)


**รูป: Actual vs Predicted (Zoomed/Alternate View)**  
ดาวน์โหลด: [xgb_fig_02_XGBoost-Forecast-vs-Actual.png](sandbox:/mnt/data/xgb_media/xgb_fig_02_XGBoost-Forecast-vs-Actual.png)  

![](xgb_media/xgb_fig_02_XGBoost-Forecast-vs-Actual.png)


**วิธีอ่านกราฟ Actual vs Predicted:**  
- เส้น/จุดของ **ค่าทำนาย** ควรเกาะติด **ค่าจริง** หากมีช่วงที่หลุดห่าง แปลว่าโมเดลตามแนวโน้มไม่ทันหรือเจอ outlier  
- ลองเพิ่มฟีเจอร์ที่จับโมเมนตัม (เช่น rolling mean/return) หรือปรับพารามิเตอร์เพื่อเก็บความผันผวนให้ดีขึ้น

---

## 5) กราฟเพิ่มเติมที่แนะนำ (เพื่อความครบถ้วนในพอร์ต)
> วางโค้ดนี้ท้ายโน้ตบุ๊กเพื่อบันทึกรูป .png เพิ่ม แล้วแปะใน README ได้เลย

```python
import os, matplotlib.pyplot as plt
os.makedirs("xgb_media", exist_ok=True)

# Feature Importance
from xgboost import plot_importance
plt.figure(figsize=(6,4))
plot_importance(model, max_num_features=20)
plt.tight_layout(); plt.savefig("xgb_media/xgb_fig_03_Feature-Importance.png", dpi=200)

# Residuals
resid = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(range(len(resid)), resid, s=10)
plt.axhline(0, linestyle="--"); plt.title("Residuals"); plt.xlabel("Index"); plt.ylabel("Residual")
plt.tight_layout(); plt.savefig("xgb_media/xgb_fig_04_Residuals.png", dpi=200)

# Predicted vs Actual (Scatter)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, s=10)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn,mx],[mn,mx],"--"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted")
plt.tight_layout(); plt.savefig("xgb_media/xgb_fig_05_Actual-vs-Predicted-Scatter.png", dpi=200)
```

---

## 6) วิธีรัน (Reproducibility)
```bash
pip install xgboost scikit-learn pandas numpy matplotlib yfinance
# เปิด xgboost.ipynb แล้วรันทุกเซลล์ตามลำดับ
```
- ถ้าเป็น time-series: อย่า shuffle ขณะ split, และใช้ validation แบบเลื่อนหน้าต่างเวลา (walk-forward) เมื่อทำโมเดลจริง

---

## 7) โครงสร้างรีโปที่แนะนำ
```
.
├─ xgboost.ipynb
├─ README.md
└─ xgb_media/
   ├─ xgb_fig_01_XGBoost-Forecast-vs-Actual.png
   ├─ xgb_fig_02_XGBoost-Forecast-vs-Actual.png
   ├─ (optional) xgb_fig_03_Feature-Importance.png
   ├─ (optional) xgb_fig_04_Residuals.png
   └─ (optional) xgb_fig_05_Actual-vs-Predicted-Scatter.png
```

---

## 8) ข้อจำกัด & งานต่อยอด (Limitations & Next Steps)
- **สัญญาณเชิงเวลา**: ลองเพิ่ม lag/rolling features เพื่อให้โมเดลเห็นบริบทมากขึ้น
- **การประเมิน**: เพิ่ม R²/MAPE, และใช้ **walk-forward** แทน random split เพื่อความสมจริง
- **Explainability**: เพิ่ม **Feature Importance/SHAP** เพื่อสื่อสารเหตุผลเชิงธุรกิจ
- **Baseline เปรียบเทียบ**: เทียบกับ **ARIMA/LSTM** เพื่อชูจุดเด่นของ XGBoost

_อัปเดตล่าสุด: 2025-10-07_
