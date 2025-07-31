# run คำสั่่ง pyinstaller --onefile --windowed GUI.py  ที่ cmd เพื่อแปลงเป็นไฟล์ .exe เพื่อให้คนที่ไม่มีความรู้ด้าน Coding สามารถใช้งานได้
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

def import_excel():
    filepath = filedialog.askopenfilename(
        title='เลือกไฟล์ Excel ที่จะนำเข้า',
        filetypes=[('Excel Files', '*.xlsx *.xls')]
    )
    return filepath

def save_excel(df_new):
    save_path = filedialog.asksaveasfilename(
        defaultextension='.xlsx',
        filetypes=[('Excel Files', '*.xlsx')],
        title='บันทึกไฟล์เป็น'
    )
    if save_path:
        df_new.to_excel(save_path, index=False, engine='openpyxl')
        messagebox.showinfo("สำเร็จ", f"บันทึกไฟล์สำเร็จที่:\n{save_path}")

def process_data(xls_path):
    try:
        # โหลดไฟล์
        xls = pd.ExcelFile(xls_path)
        sheet_names = xls.sheet_names

        if 'Order' not in sheet_names or 'Order Product' not in sheet_names:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่พบชีท 'Order' หรือ 'Order Product'\n\nชีทที่มี: {sheet_names}")
            return

        order = pd.read_excel(xls_path, sheet_name='Order')
        order_product = pd.read_excel(xls_path, sheet_name='Order Product')

        order.columns = order.columns.str.strip()
        order_product.columns = order_product.columns.str.strip()

        merged = pd.merge(order_product, order, on='Order ID', how='outer', suffixes=('_product', '_order'))

        all_cols = set(order.columns) | set(order_product.columns)

        for col in all_cols:
            if col != 'Order ID':
                col_order = col + '_order'
                col_product = col + '_product'
                if col_order in merged.columns and col_product in merged.columns:
                    merged[col] = merged[col_order].combine_first(merged[col_product])
                elif col_order in merged.columns:
                    merged[col] = merged[col_order]
                elif col_product in merged.columns:
                    merged[col] = merged[col_product]

        final_columns = ['Order ID'] + [col for col in all_cols if col != 'Order ID']
        final = merged[final_columns].copy()

        if 'วันเวลาที่สั่งซื้อ' in final.columns:
            final['วันเวลาที่สั่งซื้อ'] = pd.to_datetime(final['วันเวลาที่สั่งซื้อ'], errors='coerce')
            final['วันที่ทำการสั่งซื้อ'] = final['วันเวลาที่สั่งซื้อ'].dt.strftime('%Y-%m-%d')

        if 'เบอร์ผู้รับ' in final.columns:
            final['เบอร์ผู้รับ'] = final['เบอร์ผู้รับ'].astype(str).str.replace('-', '').str.replace(' ', '')

        if 'จำนวนชิ้น' in final.columns and 'ค่าสินค้า' in final.columns:
            final['จำนวนชิ้น'] = pd.to_numeric(final['จำนวนชิ้น'], errors='coerce').fillna(0).astype(int)
            final['ค่าสินค้า'] = pd.to_numeric(final['ค่าสินค้า'], errors='coerce').fillna(0)
            final['total'] = final['จำนวนชิ้น'] * final['ค่าสินค้า']

        # สร้างข้อมูลใหม่ตามโครงสร้าง
        data = {
            "หมายเลขคำสั่งซื้อ": final['Order ID'],
            "สถานะการสั่งซื้อ": final['สถานะ'],
            "ชื่อผู้ใช้ (ผู้ซื้อ)": final['ชื่อผู้สั่งซื้อ'],
            "วันที่ทำการสั่งซื้อ": final['วันที่ทำการสั่งซื้อ'],
            "เลขอ้างอิง SKU (SKU Reference No.)": final['รหัสสินค้า'],
            "ราคาขาย": final['ราคาต่อหน่วย'],
            "จำนวน": final['จำนวน'],
            "ราคาขายสุทธิ": final['ราคารวม'],
            "ชื่อผู้รับ": final['ชื่อผู้รับ'],
            "ที่อยู่ในการจัดส่ง": final['ที่อยู่ผู้รับ'],
            "โค้ดส่วนลดชำระโดยผู้ขาย" : final['ส่วนลด'],
            "ค่าจัดส่งที่ชำระโดยผู้ซื้อ" : final['ค่าส่ง']
        }

        df_new = pd.DataFrame(data)

        # เปิด Save As
        save_excel(df_new)

    except Exception as e:
        messagebox.showerror("เกิดข้อผิดพลาด", f" เกิดข้อผิดพลาด: {str(e)}")

def run_app():
    def on_import_click():
        filepath = import_excel()
        if filepath:
            process_data(filepath)

    root = tk.Tk()
    root.title("Excel Importer + Save As")
    root.geometry("400x200")

    import_button = tk.Button(root, text="เลือกไฟล์ Excel และแปลงข้อมูล", command=on_import_click, font=('Arial', 14))
    import_button.pack(expand=True)

    root.mainloop()

if __name__ == "__main__":
    run_app()
