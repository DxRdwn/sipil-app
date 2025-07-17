import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from openpyxl.styles import PatternFill
from openpyxl import load_workbook

app = Flask(__name__)

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Definisi kategori
kategori = {
    "Bukan Pondasi": [
        "1,2D+1,6L", "1,4D", "1.2D+Ev+Eh+L(x)rs", "0.9D-Ev+Eh(x)rs", 
        "1.2D+Ev+Eh+L(y)rs", "0.9D-Ev+Eh(y)rs", "1.2D+Ev+Emh+L(x)rs",
        "1.2D+Ev+Emh+L(y)rs", "0.9D-Ev+Emh(x)rs", "0.9D-Ev+Emh(y)rs"
    ],
    "Layan": ["D+L"],
    "Nominal": [
        "1.0D+0.7Ev+0.7Eh(x)rs", "1.0D+0.7Ev+0.7Eh(y)rs", 
        "1.0D+0.525Ev+0.525Eh+0.75L(x)rs", "1.0D+0.525Ev+0.525Eh+0.75L(y)rs",
        "0.6D-0.7Ev+0.7Eh(x)rs", "0.6D-0.7Ev+0.7Eh(y)rs"
    ],
    "Kuat": [
        "1.0D+0.7Ev+0.7Emh(y)rs", "1.0D+0.7Ev+0.7Emh(x)rs",
        "1.0D+0.525Ev+0.525Emh+0.75L(x)rs", "1.0D+0.525Ev+0.525Emh+0.75L(y)rs",
        "0.6D-0.7Ev+0.7Emh(x)rs", "0.6D-0.7Ev+0.7Emh(y)rs"
    ],
}

def klasifikasikan_kasus(kasus):
    for key, values in kategori.items():
        if kasus in values:
            return key
    return "Tidak Diketahui"

# Fungsi untuk mewarnai baris dengan kondisi tertentu
def highlight_rows(row):
    if row['status_klasifikasi'] == 'dipilih':
        return ['background-color: yellow'] * len(row)
    else:
        return [''] * len(row)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analisis')
def analisis():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Proses data
        try:
            df = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
            new_header = df.iloc[1]
            data_new = df.iloc[3:]
            data_new.columns = new_header
            data_new.reset_index(drop=True, inplace=True)

            # Pilih kolom relevan
            kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
            valid_columns = [col for col in kolom if col in data_new.columns]
            df2 = data_new[valid_columns].copy()
            df2['OutputCase'] = df2['OutputCase'].str.replace(' ', '', regex=False)
            df2['Kategori'] = df2['OutputCase'].apply(klasifikasikan_kasus)

            # Perhitungan tambahan
            df2['Joint'] = df2['Joint'].astype(int)
            df2['F3_Absolut'] = df2['F3'].abs()
            df2['M1_Absolut'] = df2['M1'].abs()
            df2['M2_Absolut'] = df2['M2'].abs()
            df2['F3_Max_Per_Category'] = df2.groupby(['Joint', 'Kategori'])['F3_Absolut'].transform('max')
            df2['F3_ratio'] = ((df2['F3_Absolut'] / df2['F3_Max_Per_Category']) * 100).round(2)
            
            df2 = (df2.sort_values(by=['Joint', 'Kategori', 'F3'], ascending=[True, True, False]))

            df2['M1_divider'] = df2.groupby(['Joint', 'Kategori'])['M1_Absolut'].transform('first')
            df2['M2_divider'] = df2.groupby(['Joint', 'Kategori'])['M2_Absolut'].transform('first')

            # Hitung rasio F3, M1, dan M2 terhadap nilai maksimum dalam persentase
            df2['F3_ratio'] = ((pd.to_numeric(df2['F3_Absolut'] / df2['F3_Max_Per_Category']) * 100)).round(2)
            df2['M1_ratio'] = ((pd.to_numeric(df2['M1_Absolut'] / df2['M1_divider']) * 100)).round(2)
            df2['M2_ratio'] = ((pd.to_numeric(df2['M2_Absolut'] / df2['M2_divider']) * 100)).round(2)
            # Hitung klasifikasi
            df2['Klasifikasi'] = ((df2['M1_ratio'] - 100) + (df2['M2_ratio'] - 100)) - (2 * (100 - df2['F3_ratio']))

            # Hitung max_klasifikasi
            df2['max_klasifikasi'] = df2.groupby(['Joint', 'Kategori'])['Klasifikasi'].transform('max')

            # Tambahkan status_klasifikasi
            df2['status_klasifikasi'] = df2.apply(
                lambda row: 'dipilih' if row['Kategori'] != 'Bukan Pondasi' and row['Klasifikasi'] == row['max_klasifikasi'] else '-',
                axis=1
            )

            
            df2 = df2.drop(columns=['F3_Absolut', 'M1_Absolut', 'M2_Absolut',
                        'F3_Max_Per_Category', 'M1_divider', 'M2_divider', 'max_klasifikasi'])

            df2 = df2.style.apply(highlight_rows, axis=1)

            # Simpan hasil
            output_file = os.path.join(app.config['RESULT_FOLDER'], 'Hasil_' + filename)
            df2.to_excel(output_file, index=False)

            return send_file(output_file, as_attachment=True)
        except Exception as e:
            return f"Gagal memproses file: {str(e)}"
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
