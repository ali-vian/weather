import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import timedelta
import os 

# ==============================================================================
# --- 1. KONFIGURASI JALUR & VARIABEL GLOBAL ---
# ==============================================================================

# --- A. Jalur Model Prediction Crop (Teman Anda) ---
CROP_PRED_MODEL_PATH = 'random_forest_model.pkl'
CROP_PRED_SCALER_PATH = 'scaler.pkl'

# --- B. Jalur Model Clustering (Anda) ---
MODEL_FILES_CLUSTERING = {
    "bangkalan": "kmeans_model_bangkalan.joblib",
    "sampang" : "kmeans_model_sampang.joblib",
    "pamekasan": "kmeans_model_pamekasan.joblib",
    "sumenep": "kmeans_model_sumenep.joblib",
    # Tambahkan kabupaten lain di sini
}
SCALER_FILES = {
    "bangkalan": "scaler_bangkalan.joblib",
    "sampang": "scaler_sampang.joblib",
    "pamekasan": "scaler_pamekasan.joblib",
    "sumenep": "scaler_sumenep.joblib",
    # Tambahkan kabupaten lain di sini
}

# --- C. Jalur Model VAR (Forecasting - Kritis) ---
# Menggunakan variabel dari kode teman Anda, namun diubah agar path tidak hardcode Windows
VAR_MODEL_PATH = "var_model_multivariate.joblib" 

# --- D. Mapping Data CSV ---
# Ganti dengan path CSV yang AKTUAL di server Anda
CSV_MAP = {
    "Bangkalan": "static/bangkalan.csv", 
    "Sampang": "static/sampang.csv",
    "Pamekasan": "static/pamekasan.csv",
    "Sumenep": "static/sumenep.csv",
}

# --- E. Variabel Global untuk Model yang Dimuat (Termasuk Model Teman Anda) ---
LOADED_MODELS_CLUSTERING = {}
LOADED_SCALERS = {} 
VAR_MODEL = None # Variabel global baru untuk model VAR
CROP_MODEL = None # Model Random Forest
CROP_SCALER = None # Scaler Crop Prediction

SUPPORTED_KABUPATEN_CLUSTERING = list(MODEL_FILES_CLUSTERING.keys())
FEATURE_NAMES_CROP = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ==============================================================================
# --- 2. FUNGSI LOADING SEMUA ASET (Berjalan saat Startup) ---
# ==============================================================================

def load_all_assets():
    """Memuat semua model (Clustering, Forecasting, Crop Prediction) ke memori."""
    global LOADED_MODELS_CLUSTERING, LOADED_SCALERS, VAR_MODEL, CROP_MODEL, CROP_SCALER
    print("--- MEMUAT SEMUA ASET ---")

    # A. Muat Model Crop Prediction (Teman Anda)
    try:
        CROP_MODEL = joblib.load(CROP_PRED_MODEL_PATH)
        CROP_SCALER = joblib.load(CROP_PRED_SCALER_PATH)
        print(f"✅ Model CROP ({CROP_PRED_MODEL_PATH}) dan Scaler berhasil dimuat.")
    except Exception as e:
        print(f"❌ GAGAL memuat model CROP: {e}")
        
    # B. Muat Model Clustering (Anda)
    for kab, filename in MODEL_FILES_CLUSTERING.items():
        try:
            model_loaded = joblib.load(filename)
            LOADED_MODELS_CLUSTERING[kab] = model_loaded
            print(f"✅ Model CLUSTERING {kab.title()} ({filename}) berhasil dimuat.")
        except Exception as e:
            print(f"❌ GAGAL memuat model CLUSTERING {kab.title()} ({filename}): {e}")
            LOADED_MODELS_CLUSTERING[kab] = None 

    # C. Muat Scaler Clustering (Anda)
    for kab, filename in SCALER_FILES.items():
        try:
            scaler_loaded = joblib.load(filename)
            LOADED_SCALERS[kab] = scaler_loaded
            print(f"✅ Scaler CLUSTERING {kab.title()} ({filename}) berhasil dimuat.")
        except Exception as e:
            print(f"❌ GAGAL memuat scaler CLUSTERING {kab.title()} ({filename}): {e}")
            LOADED_SCALERS[kab] = None 

    # D. Muat Model VAR/Forecasting (KRITIS: Perbaikan Kode Teman Anda)
    try:
        VAR_MODEL = joblib.load(VAR_MODEL_PATH)
        print(f"✅ Model VAR ({VAR_MODEL_PATH}) berhasil dimuat.")
    except Exception as e:
        print(f"❌ GAGAL memuat model VAR: {e}. PASTIKAN PATH BENAR.")
        VAR_MODEL = None
            
    print("--- SELESAI MEMUAT SEMUA ASET ---")

# Panggil fungsi ini saat startup!
load_all_assets() 

# ==============================================================================
# --- 3. INISIALISASI FLASK & ROUTE CROP PREDICTION (Teman Anda) ---
# ==============================================================================

app = Flask(__name__)

@app.route('/')
def home():
    # Asumsi Anda memiliki template index.html
    return render_template('index.html') 

@app.route('/crop')
def crop():
    # Asumsi Anda memiliki template Crop.html
    return render_template('Crop.html', feature_names=FEATURE_NAMES_CROP, form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    # Menggunakan CROP_MODEL dan CROP_SCALER global yang sudah dimuat
    if CROP_MODEL is None or CROP_SCALER is None:
        return "Error: Model atau Scaler Crop Prediction tidak dimuat. Cek file .pkl Anda.", 500
            
    try:
        data = request.form.to_dict()
        input_features = []
        form_values = {}
            
        for name in FEATURE_NAMES_CROP:
            value = float(data[name])
            input_features.append(value)
            form_values[name] = value 
            
        input_df = pd.DataFrame([input_features], columns=FEATURE_NAMES_CROP)
        
        # Scaling Data Input
        features_scaled = CROP_SCALER.transform(input_df) 
        features_scaled = pd.DataFrame(features_scaled, columns=input_df.columns)

        prediction = CROP_MODEL.predict(features_scaled)
        output = prediction[0]
            
        # Asumsi Anda memiliki template Crop.html
        return render_template('Crop.html', 
                                prediction_text=f'Hasil Prediksi (Kelas Tanaman): {output}', 
                                feature_names=FEATURE_NAMES_CROP,
                                form_values=form_values) 
        
    except KeyError as e:
        # Asumsi Anda memiliki template Crop.html
        return render_template('Crop.html', 
                                error_message=f'Error: Input untuk fitur {str(e)} hilang. Pastikan semua kolom terisi.',
                                feature_names=FEATURE_NAMES_CROP,
                                form_values=request.form.to_dict()), 400
    except ValueError:
        # Asumsi Anda memiliki template Crop.html
        return render_template('Crop.html', 
                                error_message='Error: Semua input harus berupa angka.',
                                feature_names=FEATURE_NAMES_CROP,
                                form_values=request.form.to_dict()), 400
    except Exception as e:
        # Asumsi Anda memiliki template Crop.html
        return render_template('Crop.html', 
                                error_message=f'Terjadi error tak terduga: {str(e)}',
                                feature_names=FEATURE_NAMES_CROP,
                                form_values=request.form.to_dict()), 500

# ==============================================================================
# --- 4. ROUTE FORECASTING PER KABUPATEN (Perbaikan Kode Teman Anda) ---
# ==============================================================================


@app.route('/forecast/<kabupaten>')
def forecast(kabupaten):
    try:
        csv_map = {
            "Bangkalan": r"static\bangkalan.csv",
            "Sampang": r"static\sampang.csv",
            "Pamekasan": r"static\pamekasan.csv",
            "Sumenep": r"static\sumenep.csv"
        }

        model_path = r"static\var_model_multivariate.joblib"

        if kabupaten not in csv_map:
            return jsonify({"error": "Kabupaten tidak ditemukan"}), 404

        # =====================
        # BACA CSV
        # =====================
        df = pd.read_csv(csv_map[kabupaten])

        # =====================
        # SET INDEX WAKTU
        # =====================
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # =====================
        # LOAD MODEL VAR
        # =====================
        var_model = joblib.load(model_path)

        # =====================
        # FEATURE ENGINEERING (SESUAI CSV ANDA)
        # =====================
        df['daily_temp'] = df['temp']                          # dari temp
        df['daily_humidity_diff'] = df['humidity'].diff()     # dari humidity
        df['daily_precipprob_diff'] = df['precipprob'].diff() # dari precipprob
        df = df.dropna()

        # =====================
        # FILTER FITUR SESUAI MODEL VAR
        # =====================
        model_features = [
            'daily_temp',
            'daily_humidity_diff',
            'daily_precipprob_diff'
        ]

        df = df[model_features]

        # =====================
        # AMBIL 3 HARI TERAKHIR
        # =====================
        last_3_days = df.tail(3)

        # =====================
        # FORECAST 5 HARI
        # =====================
        forecast_values = var_model.forecast(df.values, steps=5)

        forecast_dates = [
            df.index[-1] + pd.Timedelta(days=i)
            for i in range(1, 6)
        ]

        forecast_df = pd.DataFrame(
            forecast_values,
            columns=df.columns,
            index=forecast_dates
        )

        return jsonify({
            "last_days": {
                "dates": last_3_days.index.strftime('%Y-%m-%d').tolist(),
                "values": last_3_days.values.tolist()
            },
            "forecast": {
                "dates": forecast_df.index.strftime('%Y-%m-%d').tolist(),
                "values": forecast_df.values.tolist()
            },
            "columns": df.columns.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ==============================================================================
# --- 5. ROUTE CLUSTERING PER KABUPATEN (Kode Anda) ---
# ==============================================================================

@app.route('/clustering/<kabupaten>')
def get_clustering_data(kabupaten):
    # Logika Clustering Anda (Sudah benar dan menggunakan scaling)
    try:
        # PENGAMBILAN INPUT USER
        tgl = int(request.args.get("tgl", 1))
        bln = int(request.args.get("bln", 1))
        tahun = int(request.args.get("tahun", 2024))
        
        kabupaten_lower = kabupaten.lower()
        kabupaten_title = kabupaten.title()

        if kabupaten_title not in CSV_MAP:
            return jsonify({"error": "Kabupaten tidak valid"}), 400

        # AMBIL MODEL DAN SCALER DARI MEMORI
        model_kmeans = LOADED_MODELS_CLUSTERING.get(kabupaten_lower)
        scaler = LOADED_SCALERS.get(kabupaten_lower) 
        
        if model_kmeans is None:
            return jsonify({"error": f"Model clustering untuk {kabupaten_title} tidak ditemukan/gagal dimuat."}), 500
        
        if scaler is None:
             return jsonify({"error": f"Scaler untuk {kabupaten_title} tidak ditemukan/gagal dimuat. Tidak dapat melakukan scaling."}), 500


        # BACA CSV
        try:
            df = pd.read_csv(CSV_MAP[kabupaten_title])
        except FileNotFoundError:
            return jsonify({"error": f"Gagal memuat CSV: File data untuk {kabupaten_title} tidak ditemukan."}), 500
        
        df["datetime"] = pd.to_datetime(df["datetime"])

        # FILTER TANGGAL
        selected = df[
            (df["datetime"].dt.day == tgl) &
            (df["datetime"].dt.month == bln) &
            (df["datetime"].dt.year == tahun)
        ]

        if selected.empty:
            return jsonify({"error": f"Data cuaca untuk {tgl}/{bln}/{tahun} di {kabupaten_title} tidak ditemukan"}), 404

        # Daftar 5 Fitur untuk Scaling
        fitur_yang_dipakai_model = [
            "tempmax", 
            "precipprob", 
            "humidity", 
            "solarradiation", 
            "windspeed" 
        ]
        
        # 1. Ambil data mentah (raw) dari CSV
        fitur_utama_raw = selected[fitur_yang_dipakai_model].values
        
        # 2. LAKUKAN SCALING
        fitur_utama_scaled = scaler.transform(fitur_utama_raw) 
        
        # 3. PREDIKSI
        cluster = int(model_kmeans.predict(fitur_utama_scaled)[0])

        return jsonify({
            "kabupaten": kabupaten_title,
            "input_features_raw": selected[fitur_yang_dipakai_model].iloc[0].to_dict(),
            "feature_names": fitur_yang_dipakai_model,
            "predicted_cluster": cluster
        })

    except KeyError as e:
        return jsonify({"error": f"Gagal memprediksi cluster: Kolom fitur hilang atau salah nama: {str(e)}. Pastikan 5 kolom fitur sudah benar."}), 500
    except ValueError as e:
        return jsonify({"error": f"Gagal memprediksi cluster: Kesalahan nilai input atau format data: {str(e)}."}), 500
    except Exception as e:
        return jsonify({"error": f"Gagal memprediksi cluster: Terjadi error tak terduga: {str(e)}"}), 500


# ==============================================================================
# --- BLOK RUNNING APLIKASI ---
# ==============================================================================

if __name__ == "__main__":
    app.run(debug=True)