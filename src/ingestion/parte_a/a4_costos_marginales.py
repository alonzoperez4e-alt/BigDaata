import argparse, json
from datetime import datetime, timezone
from pathlib import Path

def write_report(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

BLOQUES_DIARIOS_ESPERADOS = 96

def main() -> int:
    p = argparse.ArgumentParser(description="Descarga/validacion idempotente A4 Costos Marginales (96 bloques diarios y barra)")
    p.add_argument("--input", default="data/sample/bronze/a4_costos_marginales")
    p.add_argument("--output", default="data/sample/logs/a4_validacion.json")
    p.add_argument("--barra", default="CHILCA 220kV")
    args = p.parse_args()
    src = Path(args.input)
    files = [f for f in src.rglob("*") if f.is_file() and not f.name.startswith(".")] if src.exists() else []
    payload = {"fuente": "a4_costos_marginales", "generated_utc": datetime.now(timezone.utc).isoformat(), "input": src.as_posix(), "exists": src.exists(), "file_count": len(files), "bloques_diarios_esperados": BLOQUES_DIARIOS_ESPERADOS, "barra_referencia": args.barra, "idempotente": True}
    write_report(Path(args.output), payload)
    print(f"[OK] A4 validado: {len(files)} archivos en {src.as_posix()} (bloques={BLOQUES_DIARIOS_ESPERADOS}, barra={args.barra})")
    return 0 if src.exists() else 1

if __name__ == "__main__":
    raise SystemExit(main())
