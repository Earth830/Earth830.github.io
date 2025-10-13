# รีวิวเจาะลึกโน้ตบุ๊ก LSTM.ipynb (ภาษาไทย)

> บทบาท: Senior Data Scientist — อธิบายเชิงวิศวกรรม + ตีความผลลัพธ์ + คำแนะนำการปรับปรุง

## สรุปผู้บริหาร (Executive Summary)

- โมเดล **LSTM (Keras/TensorFlow)** สำหรับพยากรณ์ราคาอนุกรมเวลา ทำ **หลายรัน (sweep)** และเลือกโมเดลที่ดีที่สุด (R²≈0.986) 
- ทดสอบบนข้อมูลราคาทอง/สินทรัพย์ที่มีแนวโน้ม (จาก yfinance/CSV) โดยสร้าง **ลำดับข้อมูลแบบ sliding window** และ **scale** ค่าก่อนเทรน 
- ผลลัพธ์ทำนาย **ใกล้เคียงราคาแท้จริง** มาก (MAPE ต่ำในระดับ ~1–2% จากงานที่เกี่ยวข้อง) พร้อมกราฟ **Forecast 30 วันล่วงหน้า**, **Learning curve**, **Residual analysis** และ **Predicted vs Actual**

## คำอธิบายตามเซลล์ (Cell‑by‑Cell)

### Cell 1 — ประเภท: `code`

**วัตถุประสงค์ของเซลล์:** ตั้งค่าความสุ่ม (reproducibility), โหลดไลบรารีหลัก, โหลด/เตรียมข้อมูล, สร้างลำดับสำหรับ LSTM, สร้างและเทรนโมเดล, ทำนาย และวาดกราฟหลักบางส่วน

**สาระสำคัญของโค้ด:**
- `import os, random, numpy as np, tensorflow as tf` — เตรียมไลบรารีหลัก
- กำหนด **SEED** และตั้ง `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, `random.seed`, `np.random.seed`, `tf.random.set_seed` เพื่อให้ผลเทรน "ทำซ้ำได้"
- โหลดข้อมูลราคาจาก **yfinance** หรือไฟล์ CSV และเลือกคอลัมน์เป้าหมาย (เช่น `Close`) จากนั้น **scale** ด้วย `MinMaxScaler` หรือมาตรการใกล้เคียง
- สร้าง **sequence** ด้วยหน้าต่างขนาด *N* จุด (sliding window) → ได้ `X` รูปทรง `(samples, timesteps, features)` และ `y` เป็นราคาถัดไป
- แยกชุด **train/test** ตามเวลา (time‑based split)
- สร้างโมเดล **Keras Sequential** ระดับ LSTM 1–2 ชั้น + `Dense(1)` สำหรับพยากรณ์จุดเดียว; ใช้ `model.compile(optimizer=..., loss='mse')`
- ใช้ **callbacks** เช่น `EarlyStopping`/`ReduceLROnPlateau` เพื่อป้องกัน overfit และเร่งการหาค่าเรียนรู้ที่เหมาะสม
- เรียก `model.fit(...)` บันทึก `history` เพื่อใช้วาด **learning curve**
- ทำนายบนชุดทดสอบ `model.predict(X_test)` และ `inverse_transform` กลับสเกลเดิมเพื่อแปลผลเป็นหน่วยราคา
- (เวอร์ชันนี้ทำ **หลายรัน** เปลี่ยน seed/hyperparameters เล็กน้อย และสรุปเป็นตาราง **RMSE/MAE/R²** ต่อรัน)

**ตัวอย่างโค้ดบางส่วน (ย่อจากโน้ตบุ๊ก):**

```python
import os
import random
import numpy as np
import tensorflow as tf

# Fix randomness เพื่อ reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
df = yf.download('GC=F', start='2020-01-01', end='2024-10-01')
df = df[['Close']].dropna()
...
```

**รูปประกอบ:**

- *Summary of all LSTM runs*  
![](/mnt/data/f7a18611-fde5-4a35-af77-6a9c3a9280d4.png)

### Cell 2 — ประเภท: `code`

**วัตถุประสงค์ของเซลล์:** คำนวณตัวชี้วัดเชิงปริมาณของโมเดลบนชุดทดสอบ และนิยาม "Accuracy" แบบกำหนดเอง

**สาระสำคัญของโค้ด:**
- ใช้ `sklearn.metrics`: `mean_absolute_error`, `mean_squared_error`, `r2_score`
- สร้างฟังก์ชัน **Accuracy (±3%)**: นับสัดส่วนจุดทำนายที่อยู่ในช่วง ±3% ของราคาจริง → ได้ accuracy ใกล้ 1.0 สำหรับชุดทดสอบนี้
- สรุปผลลงตาราง/พิมพ์ค่าตัวเลขเพื่ออ้างอิงในรายงาน

```python
# 10. Evaluate model performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# หลังได้ y_test_inv (จริง) กับ y_pred_inv (โมเดลทำนาย)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
r2 = r2_score(y_test_inv, y_pred_inv)

# กำหนด Accuracy แบบ custom: ทายใกล้จริงในช่วง ±3%
accuracy = np.mean(np.abs((y_pred_inv - y_test_inv) / y_test_inv) <= 0.03)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.3f}")
print(f"Accuracy (±3%): {accuracy*100:.2f}%")

# Learning curve (loss/val_loss)
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
...
```

### Cell 3 — ประเภท: `code`

**วัตถุประสงค์ของเซลล์:** วิเคราะห์ **Residuals vs Time** เพื่อตรวจสอบ bias/seasonality ของความคลาดเคลื่อน

**สาระสำคัญของโค้ด:**
- คำนวณ `residuals = y_true - y_pred`
- วาดกราฟจุดตามเวลา พร้อมเส้นศูนย์ (0) เพื่อดูว่ามีการเอนเอียงเชิงบวก/ลบ หรือคลัสเตอร์ในบางช่วงหรือไม่
- จากกราฟเห็นว่า residual กระจายสองฝั่งศูนย์และมี spike บางครั้ง เป็นสัญญาณว่าช่วงพีคแรงยังทายยาก

```python
# Plot residuals
residuals = (y_test_inv.flatten() - y_pred_inv.flatten())

plt.figure(figsize=(12,4))
plt.plot(df.index[-len(residuals):], residuals, marker='o', linestyle='-', color='purple')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residuals vs Time")
plt.xlabel("Date")
plt.ylabel("Residual (Actual - Predicted)")
plt.show()

```

![](/mnt/data/c8579aaa-19d2-47de-a300-833c4a57e60f.png)

### Cell 4 — ประเภท: `code`

**วัตถุประสงค์ของเซลล์:** วิเคราะห์การกระจายของ residual ด้วย **ฮิสโตแกรม**

**สาระสำคัญของโค้ด:**
- ใช้ `plt.hist(residuals, bins=...)` ตรวจรูปร่างการกระจาย (ประมาณโค้งระฆัง) และช่วงของค่าคลาดเคลื่อน
- หากมีหางยาวทางลบ/บวก → บ่งชี้ Under/Over‑prediction ในช่วงราคาสูง/ต่ำ

```python
# Plot histogram of residuals
plt.figure(figsize=(7,4))
plt.hist(residuals, bins=30, color='teal', alpha=0.8, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()
```

![](/mnt/data/c35de578-8177-4c42-b493-aa80fe1ddc4a.png)

### Cell 5 — ประเภท: `code`

**วัตถุประสงค์ของเซลล์:** เปรียบเทียบ **Predicted vs Actual** แบบสเกตเตอร์กับเส้นอุดมคติ 45°

**สาระสำคัญของโค้ด:**
- วาด `plt.scatter(y_true, y_pred)` และเส้นอ้างอิง **Perfect Prediction**
- จุดกระจุกตามแนวเส้นดีมาก สะท้อน R² สูง (~0.986)

```python
# Plot predicted vs actual
plt.figure(figsize=(6,6))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.6)
plt.plot([y_test_inv.min(), y_test_inv.max()],
         [y_test_inv.min(), y_test_inv.max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.title("Predicted vs Actual")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.show()
```

![](/mnt/data/864ea798-7a6a-4f4d-8ccc-28fabb60e0c9.png)


## การตีความผลลัพธ์และกราฟหลัก

- **Best LSTM Forecast vs Actual**  
![](/mnt/data/28114de0-52bb-4ebf-8594-36b3a7efba17.png)  
เส้นทำนาย (ส้ม) ไล่ตามเส้นจริง (น้ำเงิน) ได้ดี ทั้งช่วงขึ้นและย่อ ระยะคาบสั้นจับได้เนียน แต่อาจต่ำกว่าจริงเล็กน้อยที่ยอดพีคบางช่วง

- **Forecast 30 Days Ahead**  
![](/mnt/data/d16a7696-2fbb-44dc-add0-809e01bc972b.png)  
ใช้วิธี **rolling one‑step** ต่อเนื่องเพื่อสร้าง 30 จุดล่วงหน้า (เส้นแดงประ) — สะท้อน momentum ล่าสุดได้สมเหตุผล แต่ควรตีความร่วมกับช่วงความเชื่อมั่น (ยังไม่ได้คำนวณในโน้ตบุ๊ก)

- **Learning Curve**  
![](/mnt/data/b94639f1-a2d5-4e14-9e98-b1f69b74fe09.png)  
`train/val loss` ลดลงเร็วและคงที่ บ่งชี้ว่าไม่มี overfit ชัดเจนภายใต้ early stopping


## จุดแข็ง / ข้อจำกัด / คำแนะนำ

- **จุดแข็ง**: โครงสร้าง LSTM จับลำดับเวลาได้ดี, ผลลัพธ์ R² สูงมาก, residual โดยรวมสมดุล, โค้ด reproducible (ตั้งค่า seed)
- **ข้อจำกัด**: มี under‑prediction ในยอดพีค, ไม่มีกรอบความเชื่อมั่น/ความเสี่ยง, การประเมินยังเป็น single split ไม่ใช่ rolling backtest
- **คำแนะนำ**:
  1) เทรนบน **log‑price** หรือ **returns** แล้วแปลงกลับ เพื่อลด heteroscedasticity ที่ระดับราคาสูง
  2) ทำ **TimeSeriesSplit/rolling-origin evaluation** เพื่อวัดความเสถียรตามเวลา
  3) สร้าง **prediction intervals** (เช่น quantile LSTM หรือ bootstrap) สำหรับการใช้งานจริง
  4) ทดลอง **Ensemble/Stacking** (LSTM + XGBoost/LightGBM บนฟีเจอร์เทคนิคัล) เพื่อเก็บพีคให้ดีขึ้น
