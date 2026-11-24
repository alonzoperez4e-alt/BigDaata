import argparse, json
from datetime import datetime, timezone
from pathlib import Path

def write_report(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def main() -> int:
    p = argparse.ArgumentParser(description="Validacion/adquisicion A1 Despacho Programado")
    p.add_argument("--input", default="data/sample/bronze/a1_despacho_programado")
    p.add_argument("--output", default="data/sample/logs/a1_validacion.json")
    args = p.parse_args()
    src = Path(args.input)
    files = [f for f in src.rglob("*") if f.is_file() and not f.name.startswith(".")] if src.exists() else []
    payload = {"fuente": "a1_despacho_programado", "generated_utc": datetime.now(timezone.utc).isoformat(), "input": src.as_posix(), "exists": src.exists(), "file_count": len(files)}
    write_report(Path(args.output), payload)
    print(f"[OK] A1 validado: {len(files)} archivos en {src.as_posix()}")
    return 0 if src.exists() else 1

if __name__ == "__main__":
    raise SystemExit(main())
