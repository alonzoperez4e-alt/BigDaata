# COES SEIN Big Data - Pipeline Fase 1

## 1. Prerrequisitos exactos
- SO / shell: Windows 10/11 (Git Bash / WSL2) o Linux con bash.
- Java: OpenJDK 17 (JAVA_HOME y HADOOP_HOME / winutils en Windows si aplica).
- Python: Python 3.10.x con pyspark==3.5.1 unificado (evitar Python 3.14 o Spark 4.0.0).

## 2. Instalacion en 3 comandos
1) python3.10 -m venv .venv
2) source .venv/bin/activate
3) pip install -r requirements.txt

## 3. Orden exacto de ejecucion de scripts
1) Validar scripts de ingesta A1-A4 (src/ingestion/parte_a/) y B1-B4 (src/ingestion/parte_b/), luego generar/verificar data/manifests/bronze_sha256.json con scripts/generar_manifiesto.py y scripts/verificar_manifiesto.py.
2) Ejecutar ./run_pipeline.sh sobre data/sample (manifiesto -> ETL silver y gold -> entrenamiento_modelo.py).
3) Ejecutar pruebas de escalabilidad en dos invocaciones limpias de JVM con local_scale_100x (--stage silver y luego --stage gold) guardando los JSON en evidencias/escalabilidad/.

## 4. Salida esperada
- Integridad Bronze: OK: 8 archivos verificados con SHA-256.
- ETL: reportes JSON de silver y gold con conteos, calidad y tiempos.
- ML: metricas (TN, FP, FN, TP), importancia de variables y evidencias_modelo/fig_4_2_matriz_confusion_baseline.png.
