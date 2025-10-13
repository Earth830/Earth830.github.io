# รีวิวเชิงลึก DL_BERT.ipynb (ภาษาไทย)

> โฟกัส: งานจำแนกข้อความภาษาไทย 2 โจทย์ — **อารมณ์ความรู้สึก (3 คลาส)** และ **หัวข้อสนทนา (17 คลาส)** ด้วยโมเดลฐาน **WangchanBERTa** จาก AIResearch (`airesearch/wangchanberta-base-att-spm-uncased`).

## 1) วัตถุประสงค์

- สร้าง **Pipeline BERT** สำหรับภาษาไทยที่พร้อมใช้งานจริง: เตรียมข้อมูล → เข้ารหัส (tokenize) → เทรน/ปรับจูน → ประเมินผล → **รันทำนายแบบ batch จากไฟล์ Excel หลายชีต** และส่งออกผลลัพธ์กลับเป็น Excel.

- โจทย์ที่รองรับ:

  - **Sentiment**: `Positive / Neutral / Negative` (**num_labels=3**) ใช้ checkpoint `model_sentiment/checkpoint-1000`.

  - **Topic**: 17 หัวข้อ: Activity, After-Service, Appreciation, Chinese Investors, Common Area - Facilities, Construction Materials, Design, Engaging, Financial & Branding, Intention, Location, Pet, Politics (delete after), Price & Promotion, Quality, Security, Space (**num_labels=17**) ใช้ checkpoint `model_topic/checkpoint-1200`.

## 2) อธิบายหลักการทำงานแต่ละเซลล์

### Cell 1 — ประเภท `code`

- **บทบาท:** นำเข้าไลบรารีหลัก (`torch`, `transformers`, `pandas`, `pickle`, `logging`) สร้างตัวช่วย `load_pickle`/ตรวจไฟล์ และเตรียม **เส้นทางอินพุต/เอาต์พุต** เช่น `input_excel`, `output_excel`.

- **แกนข้อมูล:** อ่าน **Excel หลายชีต** ด้วย `pd.read_excel(input_excel, sheet_name=None)` → ได้ `dict[str, DataFrame]` สำหรับวนลูปประมวลผลแต่ละชีต.

**ตัวอย่างโค้ด (ย่อ):**
```python
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

input_excel = r"C:\othai\ML_BERT\ML_Excel\externn.xlsx"
output_dir  = r"C:\othai\ML_BERT\Model\pk2"
os.makedirs(output_dir, exist_ok=True)

topic_list = [
    "Activity", "After-Service", "Appreciation", "Chinese Investors",
    "Common Area - Facilities", "Construction Materials", "Design", "Engaging",
    "Financial & Branding", "Intention", "Location", "Pet",
    "Politics (delete after)", "Price & Promotion", "Quality", "Security", "Space"
]
topic_map  = {t: i for i, t in enumerate(topic_list)}
sentiment_map = {"Positive": 0, "Neutral": 1, "Negative": 2}

```

### Cell 2 — ประเภท `code`

- **บทบาท:** สร้าง/โหลด **Tokenizer** จากชื่อโมเดลฐาน (`AutoTokenizer.from_pretrained(model_name, use_fast=True)`) และเตรียมพารามิเตอร์ เช่น `max_length`.

- **ผลลัพธ์:** ได้ตัวเข้ารหัสข้อความให้เป็น IDs/attention mask พร้อมตัด/เติม pad ตามความยาวที่กำหนด.

**ตัวอย่างโค้ด (ย่อ):**
```python
import argparse
import logging
import os
import pickle
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR          = r"C:\othai\ML_BERT\Model\pk2"
DEFAULT_TRAIN     = os.path.join(DATA_DIR, "train_texts.pkl")
DEFAULT_TEST      = os.path.join(DATA_DIR, "test_texts.pkl")
OUTPUT_DIR        = os.path.join(DATA_DIR, "encodings")
DEFAULT_TRAIN_OUT = os.path.join(OUTPUT_DIR, "train_encodings.pkl")
DEFAULT_TEST_OUT  = os.path.join(OUTPUT_DIR, "test_encodings.pkl")
```

### Cell 3 — ประเภท `code`

- **บทบาท:** กำหนด **Mapping ป้ายกำกับ**: `sentiment_map = {"Positive":0, "Neutral":1, "Negative":2}` และ `topic_map` ที่สร้างจาก `topic_list` 17 รายการ.

- **คลาสข้อมูล:** `TextDataset` (สำหรับ inference จากลิสต์ข้อความ) และ `CommentDataset` (จับคู่ `encodings` กับ `labels` ใช้ตอนเทรน/วัดผล).

**ตัวอย่างโค้ด (ย่อ):**
```python
import os
import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle(path: str):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded {len(data)} items from {path}")
    return data
```

### Cell 4 — ประเภท `code`

- **บทบาท:** นิยามฟังก์ชัน **`get_preds(model, tokenizer, texts, ...)`** — รับลิสต์ข้อความ → สร้าง `DataLoader` → tokenize เป็น batch → ส่งผ่านโมเดลบน **GPU/CPU** → แปลง `logits` เป็น `pred_ids` และแม็ปกลับเป็น **label ชื่อ** ตาม `label_map`.

- **รายละเอียดสำคัญ:** เลือก `num_workers` และ `pin_memory` อัตโนมัติตามระบบปฏิบัติการเพื่อให้ inference เร็ว.

**ตัวอย่างโค้ด (ย่อ):**
```python
import os
import pickle
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle(path: str):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded {len(data)} items from {path}")
    return data
```

### Cell 5 — ประเภท `code`

- **บทบาท:** สร้าง/โหลดโมเดล **Sequence Classification** ด้วย `AutoModelForSequenceClassification.from_pretrained(...)` สำหรับ **Sentiment** และ **Topic** อย่างละตัว.

- **เทรน/ประเมิน:** กำหนด `TrainingArguments` (เช่น `learning_rate=2e-5`, `num_train_epochs=6`, `batch_size=16`, `weight_decay=0.02`) แล้วผูกเข้ากับ `Trainer` และเรียก `trainer.train()` / `trainer.evaluate()` โดยมี `compute_metrics` (คำนวณ **accuracy**) เพื่อรายงานผล.

**ตัวอย่างโค้ด (ย่อ):**
```python
import os
import pickle
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle(path: str):
```

### Cell 6 — ประเภท `code`

- **บทบาท:** **Batch Inference บน Excel** — วนทุกชีตของไฟล์อินพุต → เลือกคอลัมน์ข้อความ (เช่น `comment`/`message`) → เรียก `get_preds` ของ **สองโมเดล** แล้วเพิ่มคอลัมน์ `sentiment_pred` และ `topic_pred` ลงใน DataFrame.

**ตัวอย่างโค้ด (ย่อ):**
```python
import os
import pickle
import logging
import torch
import numpy as np
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

logging.basicConfig(level=logging.INFO)
```

### Cell 7 — ประเภท `code`

- **บทบาท:** **ส่งออกผลลัพธ์** ด้วย `pd.ExcelWriter(output_excel)` เขียนทุกชีตลงไฟล์ผลลัพธ์ และโค้ด `if __name__ == "__main__"` เพื่อให้รันทันทีแบบสคริปต์ (รองรับพารามิเตอร์บรรทัดคำสั่งผ่าน `argparse`).

**ตัวอย่างโค้ด (ย่อ):**
```python
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def get_preds(model, tokenizer, texts, device, batch_size=32, max_length=128, label_map=None):
    dataset = TextDataset(texts)
    def collate_fn(batch):
```

## 3) ผลลัพธ์

- หลังเทรน/ประเมินผ่าน `Trainer.evaluate()` โมเดลคำนวณ **Accuracy** บนชุดวัดผล (ค่าที่ได้ขึ้นกับข้อมูลที่ใช้)

- งาน **Inference** บนไฟล์ Excel: ได้ไฟล์ผลลัพธ์ที่มีคอลัมน์เพิ่ม `sentiment_pred` และ `topic_pred` สำหรับทุกชีต — พร้อมใช้งานต่อใน BI/แดชบอร์ดหรือส่งทีมธุรกิจ

- เช็คพอยต์ที่ใช้/เซฟ: `model_sentiment/checkpoint-1000`, `model_topic/checkpoint-1200`, และสามารถ `save_pretrained(output_dir)` เพื่อจัดเก็บเป็นอาร์ติแฟกต์.

## 4) แนวทางพัฒนาต่อยอด

1. **เพิ่มเมตริก**: นอกจาก accuracy ควรรายงาน `precision/recall/F1`, `confusion matrix` (โดยเฉพาะโจทย์ 17 คลาส) และ **macro-F1** เพื่อความยุติธรรมข้ามคลาส.

2. **Balanced Training**: ใช้ class weights หรือ focal loss หากพบคลาสไม่สมดุล โดยเฉพาะหัวข้อที่เจอน้อย.

3. **Domain Adaptation**: ทำ continued pretraining บนคอร์ปัสองค์กร (MLM) ก่อน fine-tune เพื่อยกระดับคุณภาพบนศัพท์เฉพาะ.

4. **Data Pipeline**: เพิ่มขั้นตอนทำความสะอาดข้อความไทย (ตัดอีโมจิ/ลิงก์/ซ้ำ, normalizer) และบันทึก **data version** เพื่อ reproducibility.

5. **Serving**: แพ็กเป็น **FastAPI** endpoint พร้อม batch endpoint ที่รองรับ Excel/CSV, ทำ model card และ **governance** (PII scan, consent).

6. **Monitoring**: ติดตาม drift ของสัดส่วนคลาส/ความยาวข้อความและคุณภาพ (accuracy/F1) แบบ **continual evaluation**; ใช้ **PSI/JS divergence** กับ embedding drift.

7. **Multi-task / Ensemble**: ทดลอง **multi-task head** (sentiment+topic ร่วมกัน) หรือ **ensemble checkpoints** เพื่อเพิ่มความเสถียร.
