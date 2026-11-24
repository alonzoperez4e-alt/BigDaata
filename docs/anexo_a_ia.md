# Anexo A - Declaracion de uso de IA (Fase 1)

Repositorio: https://github.com/cyonalva20/BigDaata.git

| Modulo / Tarea | Herramienta IA | Uso principal y prompt/accion | Verificacion humana |
|---|---|---|---|
| Entorno y run_pipeline.sh | Asistente IA | Estandarizacion Python 3.10 + PySpark 3.5.1 + OpenJDK 17 y ejecucion 1x | Validado en terminal con exit code 0 |
| Ingesta A1-A4 y manifiesto SHA-256 | Asistente IA | Scripts parte_a y verificacion de bronze_sha256.json | Verificado con verificar_manifiesto.py (8 archivos) |
| Escalabilidad 100x | Asistente IA | Corridas limpias silver y gold con local_scale_100x en evidencias/escalabilidad/ | Revisados run_100x_silver.json y run_100x_gold.json |
| Documentacion README y anexos | Asistente IA | Redaccion de README.md, matriz de entornos y auditoria de cierre | Revisado por el equipo antes del push a feat/fase-1 |

## Firmas de integrantes
- Fabian Joseph Cubas Torres | Codigo: ____________
- Cristhian Jhair Alonzo Perez | Codigo: ____________
- Joan Paolo Solar Beltran | Codigo: ____________
- Carlos Daniel Yon Alva | Codigo: ____________
