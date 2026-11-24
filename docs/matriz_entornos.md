# Matriz de Entornos por Experimento (Secciones 0.4, 2.2 y 3.7)

| Experimento / Etapa | Perfil / Entorno | SO / Runtime | Parametros clave | Estado / Evidencia |
|---|---|---|---|---|
| Base 1x (run_pipeline.sh) | local_dev | Python 3.10 + PySpark 3.5.1 + OpenJDK 17 | Muestra data/sample | Completado (exit code 0) |
| Escalabilidad Silver 100x | local_scale_100x | Python 3.10 + PySpark 3.5.1 + OpenJDK 17 | Invocacion JVM limpia (--stage silver) | evidencias/escalabilidad/run_100x_silver.json |
| Escalabilidad Gold 100x | local_scale_100x | Python 3.10 + PySpark 3.5.1 + OpenJDK 17 | Invocacion JVM limpia (--stage gold) | evidencias/escalabilidad/run_100x_gold.json |
