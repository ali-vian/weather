import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd

# 🛠️ Muat Model dan Scaler
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model dan Scaler berhasil dimuat.")
except FileNotFoundError:
    print("ERROR: File model/scaler tidak ditemukan. Harap simpan dulu model dan scaler Anda.")
    # Agar aplikasi tidak crash saat dijalankan tanpa file
    model, scaler = None, None 

app = Flask(__name__)

# Daftar fitur yang diharapkan (Harus SAMA dengan urutan saat training)
FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# --- Route untuk Halaman Utama (Form Input) ---

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop')
def crop():
    # Menampilkan form input kosong saat pertama kali diakses
    return render_template('Crop.html', feature_names=FEATURE_NAMES, form_values={})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
         return "Error: Model atau Scaler tidak dimuat. Cek file .pkl Anda.", 500
         
    try:
        data = request.form.to_dict()
        input_features = []
        form_values = {}
        
        # 1. Ambil data, konversi, dan validasi
        for name in FEATURE_NAMES:
            value = float(data[name]) # Akan melempar ValueError jika bukan angka
            input_features.append(value)
            form_values[name] = value 
        
        # 2. KONVERSI KE DATAFRAME DENGAN NAMA FITUR YANG BENAR
        # Ini mengatasi UserWarning
        input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES)

          # Debug: Cek DataFrame input
        
        # 3. Scaling Data Input
        # Scaler bisa menerima DataFrame atau Array.
        # Catatan: Jika scaler Anda adalah objek yang membutuhkan array (misalnya hanya mengambil values), 
        # gunakan features_scaled = scaler.transform(input_df.values)
        features_scaled = scaler.transform(input_df) 
        features_scaled = pd.DataFrame(features_scaled,columns=input_df.columns)

        print(features_scaled.head())  # Debug: Cek data setelah scaling
        # 4. Prediksi
        # Model Random Forest (Scikit-learn) tetap memerlukan array/list/DataFrame untuk prediksi
        prediction = model.predict(features_scaled)

        print(f"Prediksi: {prediction}")  # Debug: Cek hasil prediksi

        
        
        # ... (Sisa code untuk formatting output)
        output = prediction[0]
        
        return render_template('Crop.html', 
                               prediction_text=f'Hasil Prediksi (Kelas Tanaman): {output}', 
                               feature_names=FEATURE_NAMES,
                               form_values=form_values) 
    
    # ... (Penanganan Error)
    except KeyError as e:
        # Error 400 karena salah satu fitur tidak terkirim dari form
        return render_template('Crop.html', 
                               error_message=f'Error: Input untuk fitur {str(e)} hilang. Pastikan semua kolom terisi.',
                               feature_names=FEATURE_NAMES,
                               form_values=request.form.to_dict()), 400
    except ValueError:
        # Error 400 karena input tidak bisa diubah ke float
        return render_template('Crop.html', 
                               error_message='Error: Semua input harus berupa angka.',
                               feature_names=FEATURE_NAMES,
                               form_values=request.form.to_dict()), 400
    except Exception as e:
        return render_template('Crop.html', 
                               error_message=f'Terjadi error tak terduga: {str(e)}',
                               feature_names=FEATURE_NAMES,
                               form_values=request.form.to_dict()), 500

if __name__ == "__main__":
    app.run(debug=True)