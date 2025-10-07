# Auto Google Forms Filler — Selenium + YAML Quotas (TH/EN)

> **TH:** สคริปต์นี้ช่วยกรอก Google Forms อัตโนมัติโดยใช้ **Selenium** และกำหนดพฤติกรรมด้วยไฟล์ **YAML** (เช่น น้ำหนักการสุ่มคำตอบ, โควตา, รูปแบบกริด) เหมาะสำหรับงานวิจัย/สำรวจ/ทดสอบระบบ พร้อมโหมด **Dry‑run**, **Headless**, และ **ผลลัพธ์สรุป**  
> **EN:** This project auto‑fills Google Forms using **Selenium**, configured via **YAML** (weighted choices, quotas, grid formats). Includes **dry‑run**, **headless** mode, and **result summaries**.

---

## ✨ Highlights
- **YAML‑driven**: แยกโค้ดออกจากการตั้งค่า ปรับเปลี่ยนฟอร์มได้โดยไม่ต้องแก้โค้ด
- **Weighted random**: สุ่มคำตอบตามสัดส่วน (`default_weights`) เพื่อจำลองพฤติกรรมผู้ตอบ
- **Multiple Choice Grid** รองรับเต็มรูปแบบ (กำหนด `columns`, `rows`, บังคับเลือกครบทุกแถว `require_each_row`)
- **Retry & Robust selectors**: พยายามหา element หลายวิธี (label text, `aria-label`, XPath fallback)
- **Headless / Visible**: รันแบบไม่โชว์หน้าต่าง หรือเปิดหน้าต่างสำหรับ debug
- **Dry‑run**: ทดสอบ flow โดยไม่ส่งฟอร์มจริง (ป้องกันสแปม)
- **Results Log**: เก็บสรุปสถานะใน DataFrame (OK/FAIL, HTTP code เมื่อใช้โหมด requests, เวลาในการทำงาน)

---

## 📁 Project Structure
- `auto_oogle_forms3.ipynb` — Notebook หลัก (มีปุ่ม/ตัวแปรสำหรับ Run)
- `config_quota_updated.yaml` — ไฟล์ตั้งค่า/คำตอบ (ตัวอย่างสกีมาด้านล่าง)

> **หมายเหตุ:** โปรเจกต์ตั้งใจให้ใช้งานบน Notebook เพื่อความยืดหยุ่นในการ Debug และปรับค่า runtime

---

## ⚙️ Setup
```bash
# แนะนำสร้าง virtual env ก่อน
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install selenium webdriver-manager pyyaml pandas
```

> Chrome/Chromium ต้องติดตั้งไว้ในเครื่อง `webdriver-manager` จะช่วยดึงไดรเวอร์ให้อัตโนมัติ

---

## 🧩 YAML Schema (สำคัญ)
ตัวอย่างโครงสร้างที่สคริปต์รองรับ (ย่อ/หลัก):

```yaml
form:
  url: "https://docs.google.com/forms/XXXXXXXX"   # ประเภทลิงก์

run:
  n_submissions: 10 #รัน 10 ครั้ง
  headless: true
  dry_run: false
  delay_between_clicks_sec: [0.3, 0.8]   # สุ่มดีเลย์ระหว่างคลิก/พิมพ์ 
  delay_between_submissions_sec: [1.0, 2.0]

answers:
  # ===== ตัวอย่าง: ข้อเลือกเดี่ยว =====
  - type: multiple_choice
    match:
      question_contains: "เพศของคุณ"
    default_weights:
      "ชาย": 0.45
      "หญิง": 0.55

  # ===== ตัวอย่าง: กรอกข้อความ =====
  - type: short_answer
    match:
      question_contains: "ชื่อเล่น"
    value: "สมชาย"

  # ===== ตัวอย่าง: Checkbox (ติ๊กหลายข้อ) =====
  - type: checkboxes
    match:
      question_contains: "แอปที่คุณใช้เป็นประจำ"
    default_weights:
      "Grab": 0.4
      "LINE MAN": 0.35
      "Robinhood": 0.25
    min_checks: 1
    max_checks: 2

  # ===== ตัวอย่าง: Multiple Choice Grid =====
  - type: multiple_choice_grid
    match:
      question_contains: "ความพึงพอใจต่อแอปพลิเคชันบริการเรียกรถที่คุณใช้เป็นประจำ"
    columns: ["น้อยที่สุด","น้อย","ปานกลาง","มาก","มากที่สุด"]
    require_each_row: true
    # น้ำหนักค่าเริ่มต้นสำหรับทุกแถว (แก้ไขเป็นรายแถวได้)
    default_weights:
      "น้อยที่สุด": 0.05
      "น้อย": 0.15
      "ปานกลาง": 0.35
      "มาก": 0.30
      "มากที่สุด": 0.15
    # (ออปชัน) กำหนดเฉพาะเจาะจงต่อแถว
    row_overrides:
      - row_contains: "ความรวดเร็ว"
        weights:
          "ปานกลาง": 0.25
          "มาก": 0.50
          "มากที่สุด": 0.25
```

**คำอธิบายฟิลด์สำคัญ**
- `type`: ประเภทคำถามที่ฟอร์มเป็นจริง (เช่น `short_answer`, `multiple_choice`, `checkboxes`, `multiple_choice_grid`)
- `match.question_contains`: ข้อความบางส่วนจากหัวข้อคำถาม เพื่อให้สคริปต์หา block ที่ถูกต้อง
- `default_weights`: สัดส่วนสุ่มคำตอบ (ค่ารวมไม่จำเป็นต้องเท่ากับ 1 ระบบจะทำ normalization ให้)
- `require_each_row` (เฉพาะ grid): ถ้า `true` จะพยายามเลือกให้ครบทุกแถว
- `row_overrides` (ออปชัน): ปรับน้ำหนักการสุ่มเฉพาะบางแถวในกริด

---

## ▶️ How to Run (ใน Notebook)
1. เปิด `auto_oogle_forms3.ipynb`
2. ตั้งค่าตัวแปรหลัก:
   ```python
   CFG_PATH = "/path/to/config_quota_updated.yaml"
   N_SUBMISSIONS = 10      # จำนวนครั้งที่ต้องการส่ง
   HEADLESS = True         # ไม่แสดงหน้าต่างเบราว์เซอร์
   DRY_RUN = False         # True = ทดสอบไม่กดส่งจริง
   ```
3. กด Run ตามลำดับเซลล์ จนได้ตารางสรุปผล (OK/FAIL, เวลา, โหมด)

> หากต้องการ Debug UI ให้ตั้ง `HEADLESS=False` จะเห็นการคลิก/กรอกแบบ real‑time

---

## 🛠️ Implementation Notes
- **Element Locators**: พยายามระบุด้วย `aria-label` / text normalize / XPath fallback เพื่อรองรับ DOM ที่เปลี่ยนเล็กน้อย (เช่น ป้ายปุ่ม “ถัดไป/ส่ง” ที่มีช่องว่าง/ขึ้นบรรทัดใหม่)
- **Wait Strategy**: ใช้ explicit wait (presence/visibility) + ดีเลย์สุ่มเพิ่มความเสถียร/เลียนแบบมนุษย์
- **Grid Handling**:
  - หา container ของคำถามจากหัวข้อ (`question_contains`)
  - อ่านรายการแถว (rows) และคอลัมน์ (columns) ที่มองเห็น
  - สำหรับแต่ละแถว: สุ่มคอลัมน์ตาม `row_overrides` → ถ้าไม่มีก็ใช้ `default_weights`
  - คลิกเฉพาะ radio ในคอลัมน์ที่เลือก
- **Safety**: เคารพ `DRY_RUN`; มีการ guard ป้องกันการกด “ส่ง” เมื่อเปิดโหมดทดสอบ

---

## 🧪 Troubleshooting
- **กด “ถัดไป/ส่ง” ไม่ติด**: ป้ายปุ่มอาจมีช่องว่าง/บรรทัดใหม่ → ใช้ตัวเลือกค้นหาแบบยืดหยุ่น (`contains(normalize-space(), ...)`) และลอง scroll‑into‑view ก่อนคลิก
- **โดน Required เตือน**: entry id/ชนิดคำถามเปลี่ยนจากของเดิม → อัปเดต `match.question_contains` หรือ YAML ให้ตรงกับฟอร์มปัจจุบัน
- **HTTP 400 เมื่อยิงแบบ requests**: ฟอร์มหลายหน้าต้องจัดการ hidden fields (`fbzx`, `fvv`, `pageHistory`, `partialResponse`) ให้ครบ → แนะนำใช้ Selenium เป็นค่าเริ่มต้น
- **ภาษาไทย/วรรณยุกต์**: ใช้การ normalize ข้อความก่อนเทียบ
- **Anti‑Spam**: ตั้ง `delay_between_submissions_sec` ให้เหมาะสมและใช้ `DRY_RUN=True` ระหว่างทดสอบ

---

## 📜 Legal & Ethics
ใช้เพื่อการทดสอบ/วิจัยเท่านั้น หลีกเลี่ยงการสแปม/ละเมิดนโยบายของแพลตฟอร์มและกฎหมายที่เกี่ยวข้อง รับผิดชอบการใช้งานของคุณเอง

---

## 📈 Roadmap (สั้น ๆ)
- รองรับ **checkbox grid**
- โหมด **hybrid** (Selenium นำทาง, requests ส่ง)
- สถิติผลลัพธ์ + บันทึก CSV อัตโนมัติ
- ตัวช่วย **extract form schema → YAML** จากลิงก์ (กึ่งอัตโนมัติ)

---

## 🖊️ Credit
พัฒนาเพื่อโปรเจกต์งานวิจัย/พอร์ตโฟลิโอ โดยเน้นความเสถียรและความยืดหยุ่นในการใช้งานกับ Google Forms หลากหลายรูปแบบ
```
