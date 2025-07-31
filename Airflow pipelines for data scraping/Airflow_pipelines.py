from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import timezone
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np



def action():
    df = scrape()
    # เพิ่มคอลัมน์ "Image" ที่จะใช้สูตร =IMAGE() สำหรับแสดงภาพ
    df['รูปสินค้า'] = df['URL รูป'].apply(lambda x: f'=IMAGE("{x}")' if pd.notnull(x) else "")
    data = [df.columns.to_list()] + df.values.tolist()
    upload_to_sheet(data)

def scrape():
    import os
    import time
    import pandas as pd
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from bs4 import BeautifulSoup
    from datetime import datetime

    URLS = {
        "https://talaadthai.com/products?market=26": "ตลาดดอกไม้",
        "https://talaadthai.com/products?market=12": "ตลาดส้ม",
        "https://talaadthai.com/products?market=5": "ตลาดผลไม้ฤดูกาล",
        "https://talaadthai.com/products?market=11": "ตลาดผลไม้รวม",
        "https://talaadthai.com/products?market=14": "ตลาดผลไม้นานาชาติ",
        "https://talaadthai.com/products?market=13": "ตลาดเเตงโม",
        "https://talaadthai.com/products?market=10": "ตลาดผักและสมุนไพร",
        "https://talaadthai.com/products?market=21": "ตลาดพืชไร่",
        "https://talaadthai.com/products?market=8": "ตลาดสด",
        "https://talaadthai.com/products?market=30": "ตลาดเนื้อสัตว์",
        "https://talaadthai.com/products?category=15": "ตลาดปลาและอาหารทะเล",
        "https://talaadthai.com/products?market=19": "ตลาดข้าวสาร",
        "https://talaadthai.com/products?market=23": "ตลาดอาหารแห้ง",
        "https://talaadthai.com/products?market=24": "ตลาดสินค้าเบ็ดเตล็ด",
        "https://talaadthai.com/products?market=29": "ตลาดสัตว์เลี้ยง",
    }

    today = datetime.now().strftime("%Y-%m-%d")
    output_folder = f"data/{today}"
    os.makedirs(output_folder, exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    driver = webdriver.Chrome(options=chrome_options)

    all_products = []

    for url in URLS:
        print(f"🔍 กำลังโหลดข้อมูลจาก: {url}")
        driver.get(url)
        time.sleep(2)

        while True:
            try:
                load_more_button = driver.find_element(By.CSS_SELECTOR, ".loadMore div.button-children")
                driver.execute_script("arguments[0].click();", load_more_button)
                time.sleep(1.5)
            except:
                print("✅ Load More จนสุดแล้ว")
                break

        product_cards = driver.find_elements(By.CSS_SELECTOR, "div.detail-grid")
        for card in product_cards:
            try:
                driver.execute_script("arguments[0].scrollIntoView();", card)
                time.sleep(0.2)
            except:
                continue

        soup = BeautifulSoup(driver.page_source, "html.parser")
        product_cards = soup.select("div.detail-grid")

        for card in product_cards:
            product_name = card.select_one("div.productName")
            product_image = card.select_one("img")
            price_max = card.select_one("div.maxPrice")
            unit = card.select_one("div.unit")
            last_update = card.select_one("div.updateDate")

            img_src = product_image["src"] if product_image else ""
            if "/error_image.jpg" in img_src:
                img_src = "https://talaadthai.com/error_image.jpg"

            all_products.append({
                "Source URL": url,
                "ชื่อสินค้า": product_name.text.strip() if product_name else "",
                "URL รูป": img_src,
                "Price Raw": price_max.text.strip() if price_max else "",
                "หน่วย": unit.text.strip() if unit else "",
                "อัพเดทล่าสุด": last_update.text.strip() if last_update else "",
                "หมวดหมู่": URLS.get(url, "ไม่ทราบ"),
            })

    driver.quit()

    df = pd.DataFrame(all_products)
    df["Price Clean"] = df["Price Raw"].str.replace("บาท", "", regex=False).str.replace(",", "", regex=False).str.strip()
    df["ราคา"] = pd.to_numeric(df["Price Clean"], errors='coerce') * 1.05
    df["ราคา"] = df["ราคา"].round(2).apply(lambda x: f"{x:,.2f} บาท" if pd.notnull(x) else "")
    df.drop(columns=["Price Clean", "Price Raw","Source URL"], inplace=True)

    return df


def upload_to_sheet(data):
    WEB_APP_URL = 'https://script.google.com/ใส่ระหัส'
    payload = {
        "sheetName": 'Sheet',
        "data": data
    }

    response = requests.post(WEB_APP_URL, json=payload)
    if response.status_code == 200:
        print('Insert Successful')
        print("Response:", response.text)
    else:
        print('Insert Error, status code:', response.status_code)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

local_tz = timezone("Asia/Bangkok")

dag = DAG(
    'TalaadThai',
    default_args=default_args,
    description='TalaadThai',
    schedule_interval='0 17 * * *',  # เที่ยงคืนไทย = 17:00 UTC
    start_date=datetime(2025, 1, 7),
    catchup=False,
)

task = PythonOperator(
    task_id='task',
    python_callable=action,
    retries=3,
    retry_delay=timedelta(minutes=10),
    dag=dag,
)
