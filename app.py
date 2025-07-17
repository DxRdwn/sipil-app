import os
import pandas as pd
import json
from flask import Response
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from openpyxl.styles import PatternFill
from openpyxl import load_workbook
from flask_cors import CORS
from model.lstm_model import run_lstm_model
from model.dnn_model import run_dnn_model
from model.cnn_model import run_cnn_model
from functools import wraps

app = Flask(__name__)
CORS(app)

API_KEY = "apDVQys0pPwxG73ZD6Df+CRpWsjafaZUbul" 

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-KEY')
        if key != API_KEY:
            return jsonify({
                "status": False,
                "code": 401,
                "message": "Unauthorized: API Key tidak valid"
            }), 401
        return f(*args, **kwargs)
    return decorated

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

ALLOWED_EXTENSIONS = {'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({
        "status": True,
        "message": "API is healthy"
    })


@app.route('/upload-file/<id_projek>', methods=['POST'])
@require_api_key
def upload_file(id_projek):
    if 'file' not in request.files:
        return jsonify({"status": False, "message": "File tidak ditemukan"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": False, "message": "Nama file kosong"}), 400
    
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": False,
            "message": "Jenis file tidak didukung. Harus .xlsx"
        }), 400

    filename = secure_filename(f"{id_projek}.xlsx")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    return jsonify({
        "status": True,
        "code": 200,
        "message": f"File berhasil diunggah untuk ID proyek: {id_projek}"
    })


@app.route('/beban-kritis/<id_projek>', methods=['GET'])
@require_api_key
def hitung_beban_kritis(id_projek):
    try:
        filename = f"{id_projek}.xlsx"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(file_path):
            return jsonify({
                "status": False,
                "code": 404,
                "message": "File proyek tidak ditemukan"
            }), 404

        # Proses file Excel
        df = pd.read_excel(file_path, sheet_name='Joint Reactions', header=None)
        new_header = df.iloc[1]
        data_new = df.iloc[3:]
        data_new.columns = new_header
        data_new.reset_index(drop=True, inplace=True)

        kolom = ['Joint', 'OutputCase', 'CaseType', 'StepType', 'F1', 'F2', 'F3', 'M1', 'M2', 'M3']
        valid_columns = [col for col in kolom if col in data_new.columns]
        df2 = data_new[valid_columns].copy()
        
        #model
        result_lstm = run_lstm_model(df2)
        result_dnn = run_dnn_model(file_path)
        result_cnn = run_cnn_model(file_path)
        
        
        df2['OutputCase'] = df2['OutputCase'].str.replace(' ', '', regex=False)
        df2['Kategori'] = df2['OutputCase'].apply(klasifikasikan_kasus)

        df2['Joint'] = df2['Joint'].astype(int)
        df2['F3_Absolut'] = df2['F3'].abs()
        df2['M1_Absolut'] = df2['M1'].abs()
        df2['M2_Absolut'] = df2['M2'].abs()
        df2['F3_Max_Per_Category'] = df2.groupby(['Joint', 'Kategori'])['F3_Absolut'].transform('max')

        df2 = df2.sort_values(by=['Joint', 'Kategori', 'F3'], ascending=[True, True, False])
        df2['M1_divider'] = df2.groupby(['Joint', 'Kategori'])['M1_Absolut'].transform('first')
        df2['M2_divider'] = df2.groupby(['Joint', 'Kategori'])['M2_Absolut'].transform('first')

        df2['F3_ratio'] = ((df2['F3_Absolut'] / df2['F3_Max_Per_Category']) * 100).round(2)
        df2['M1_ratio'] = ((df2['M1_Absolut'] / df2['M1_divider']) * 100).round(2)
        df2['M2_ratio'] = ((df2['M2_Absolut'] / df2['M2_divider']) * 100).round(2)

        df2['Klasifikasi'] = ((df2['M1_ratio'] - 100) + (df2['M2_ratio'] - 100)) - (2 * (100 - df2['F3_ratio']))
        df2['max_klasifikasi'] = df2.groupby(['Joint', 'Kategori'])['Klasifikasi'].transform('max')

        df2['status_klasifikasi'] = df2.apply(
            lambda row: 'dipilih' if row['Kategori'] != 'Bukan Pondasi' and row['Klasifikasi'] == row['max_klasifikasi'] else '-',
            axis=1
        )

        # Simpan CSV
        output_csv_filename = f"{id_projek}_hasil.csv"
        output_csv_path = os.path.join(app.config['RESULT_FOLDER'], output_csv_filename)

        df2.drop(columns=[
            'F3_Absolut', 'M1_Absolut', 'M2_Absolut',
            'F3_Max_Per_Category', 'M1_divider', 'M2_divider', 'max_klasifikasi'
        ]).to_csv(output_csv_path, index=False)

        # Ambil baris dipilih
        data_dipilih = df2[df2['status_klasifikasi'] == 'dipilih'][['Joint', 'OutputCase', 'Kategori']]
        hasil = data_dipilih.to_dict(orient='records')

        
        
        response_data = {
            "status": True,
            "code": 200,
            "message": "Perhitungan beban kritis berhasil",
            "file_hasil_url": url_for('unduh_hasil', filename=output_csv_filename, _external=True),
            "data": hasil,
            "LSTM": result_lstm,
            "DNN" : result_dnn,
            "CNN" : result_cnn
            
        }
        return Response(json.dumps(response_data), mimetype='application/json')

    except Exception as e:
        return jsonify({
            "status": False,
            "code": 500,
            "message": f"Gagal memproses file: {str(e)}"
        }), 500

@app.route('/results/<filename>')
@require_api_key
def unduh_hasil(filename):
    full_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    print(f"Request download for: {full_path}")
    
    if not os.path.exists(full_path):
        return jsonify({
            "status": False,
            "code": 404,
            "message": "File tidak ditemukan"
        }), 404
    
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

@app.route('/hapus-file/<id_projek>', methods=['DELETE'])
def hapus_file(id_projek):
    try:
        # Tentukan path file
        file_excel = os.path.join(app.config['UPLOAD_FOLDER'], f"{id_projek}.xlsx")
        file_csv = os.path.join(app.config['RESULT_FOLDER'], f"{id_projek}_hasil.csv")

        hasil = {
            "excel_dihapus": False, 
            "csv_dihapus": False
        }

        # Hapus file excel
        if os.path.exists(file_excel):
            os.remove(file_excel)
            hasil["excel_dihapus"] = True

        # Hapus file hasil csv
        if os.path.exists(file_csv):
            os.remove(file_csv)
            hasil["csv_dihapus"] = True

        if not hasil["excel_dihapus"] and not hasil["csv_dihapus"]:
            return jsonify({
                "status": False,
                "code": 404,
                "message": "Tidak ada file yang ditemukan untuk ID proyek ini",
                "hasil": hasil
            }), 404

        return jsonify({
            "status": True,
            "code": 200,
            "message": "File berhasil dihapus",
            "hasil": hasil
        })

    except Exception as e:
        return jsonify({
            "status": False,
            "code": 500,
            "message": f"Terjadi kesalahan saat menghapus file: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
