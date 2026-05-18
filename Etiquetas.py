import os
import pandas as pd
import random

# --- CONFIGURACIÓN ---
ruta_carpeta = r'C:\DataSet\notas_clinicas_corpus_distemist\notas_clinicas_corpus_distemist' # Pon la ruta real de tu carpeta de archivos .txt
limite_archivos = 1000
nombre_salida = 'dataset_para_colab.csv'

def refined_label(text):
    t = text.lower()
    
    # --- LABEL 2: ASSESSMENT (Diagnóstico y Juicio Clínico) ---
    # Es la conclusión del médico tras analizar S y O.
    if any(k in t for k in [
        "diagnóstico", "dx:", "impresión", "juicio clínico", "sospecha", 
        "probable", "confirmado", "crónico", "agudo", "fisiopatología",
        "etiología", "análisis clínico", "diagnostico", "dx clínico"
    ]): return 2

    # --- LABEL 3: PLAN (Tratamiento e Indicaciones) ---
    # Lo que se debe hacer: medicinas, dosis, citas, estudios futuros.
    if any(k in t for k in [
        "se indica", "prescribe", "tratamiento", "cita", "instrucciones", 
        "recetar", "dosis", "comprimidos", "mg/", "ml/", "vía oral", 
        "cada 8 horas", "cada 12 horas", "cada 24 horas", "por 7 días", 
        "seguimiento", "interconsulta", "referencia", "solicitar estudios", 
        "paracetamol", "ibuprofeno", "antibiótico", "dieta", "reposo"
    ]): return 3

    # --- LABEL 1: OBJECTIVE (Hallazgos Físicos y Signos Vitales) ---
    # Lo que el médico VE o MIDE directamente.
    if any(k in t for k in [
        "exploración", "inspección", "ta:", "fc:", "fr:", "temp:", "peso:", 
        "talla:", "imc:", "mmhg", "lpm", "ºc", "auscultación", "palpación", 
        "percusión", "abdomen", "tórax", "neurológico", "consciente", 
        "orientado", "hidratado", "ruidos", "ritmo", "saturación", "o2",
        "pupilas", "reflejos", "laboratorio", "rx", "ultrasonido"
    ]): return 1

    # --- LABEL 0: SUBJECTIVE (Síntomas y Antecedentes) ---
    # Lo que el paciente cuenta (Anamnesis).
    if any(k in t for k in [
        "refiere", "manifiesta", "motivo de consulta", "antecedentes", 
        "siente", "duele", "dolor", "mareo", "náuseas", "fatiga", 
        "desde hace", "comenta", "padece", "alérgico", "alergias", 
        "fuma", "alcohol", "diabético", "hipertenso", "familiar", 
        "inicio", "evolución", "cuadro clínico", "paciente refiere"
    ]): return 0

    return -1


# 1. Leer archivos
data = []
if not os.path.exists(ruta_carpeta):
    print(f"Error: La ruta {ruta_carpeta} no existe.")
else:
    archivos = [f for f in os.listdir(ruta_carpeta) if f.endswith('.txt')]
    muestra = random.sample(archivos, min(len(archivos), limite_archivos))
    print(f"Procesando {len(muestra)} archivos...")

    for f_name in muestra:
        ruta_completa = os.path.join(ruta_carpeta, f_name)
        with open(ruta_completa, 'r', encoding='utf-8', errors='ignore') as f:
            for linea in f:
                texto = linea.strip()
                if len(texto) < 15: continue
                label = refined_label(texto)
                if label != -1:
                    data.append({"text": texto, "label": label})

    # 2. Balanceo y guardado
    if data:
        df = pd.DataFrame(data)
        # Limpieza de basura común en archivos de texto
        df = df[~df['text'].str.contains("#¿NOMBRE?")]
        
        # Balanceo: para que el modelo no tenga favoritos
        min_muestras = df['label'].value_counts().min()
        df_balanceado = df.groupby('label').head(min_muestras).sample(frac=1).reset_index(drop=True)
        
        df_balanceado.to_csv(nombre_salida, index=False, encoding='utf-8-sig')
        print(f"\n¡Éxito! Archivo '{nombre_salida}' generado.")
        print("Distribución de etiquetas:")
        print(df_balanceado['label'].value_counts())
    else:
        print("No se encontraron frases que coincidieran con las etiquetas.")