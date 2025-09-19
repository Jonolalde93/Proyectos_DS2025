import os, re, json, time, argparse
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

from config import CCAA_BOUNDING_BOXES

API_URL = "https://api.openchargemap.io/v3/poi"
HEADERS = {}  

def usagecost_to_eur_per_kwh(text: str, treat_free_as_zero: bool = False):
    """
    Devuelve (valor_float | None, motivo_str).
    - Extrae el primer precio en €/kWh (o EUR/kWh) de la cadena (varias variantes).
    - Ignora precios por minuto/sesión.
    - Maneja 'desde 0,29 €/kWh', espacios raros y 'kW·h'.
    - treat_free_as_zero: convierte 'free/gratis' en 0.0 si True.
    """
    if text is None:
        return None, "empty"

    s = re.sub(r"\s+", " ", text.strip(), flags=re.UNICODE)

    if treat_free_as_zero and re.search(r"\b(free|gratis)\b", s, flags=re.I):
        return 0.0, "free->0"

    pattern = r"([0-9]+[.,]?[0-9]*)\s*(?:€|EUR)\s*/?\s*kW?h"
    matches = re.findall(pattern, s, flags=re.I)
    if not matches:
        m2 = re.search(r"([0-9]+[.,]?[0-9]*)\s*(?:€|EUR).*kW?h", s, flags=re.I)
        if m2:
            matches = [m2.group(1)]
        else:
            if re.search(r"(€|EUR)\s*/\s*(min|minute)", s, flags=re.I):
                return None, "per_minute"
            if re.search(r"(€|EUR)\s*/\s*(session|sesión)", s, flags=re.I):
                return None, "per_session"
            if re.search(r"\b(free|gratis)\b", s, flags=re.I):
                return None, "free"
            return None, "no_kwh"

    val_str = matches[0].replace(",", ".")
    try:
        val = float(val_str)
    except ValueError:
        return None, "parse_error"

    if not (0.05 <= val <= 2.0):
        return None, "out_of_range"

    return val, "ok"

def band_from_power_kw(power_kw, level_id=None):
    """
    Clasificación simple por potencia:
      - AC: <= 22 kW
      - DC: > 22 kW
    """
    if power_kw is None:
        return None
    try:
        p = float(power_kw)
        return "AC" if p <= 22 else "DC"
    except Exception:
        return None

def ensure_dirs():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/qc").mkdir(parents=True, exist_ok=True)  # opcional: calidad/diagnóstico

def load_api_key():
    load_dotenv()
    api_key = os.getenv("OCM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No se encontró OCM_API_KEY en .env")
    return api_key

def fetch_ccaa(bbox: dict, api_key: str, maxresults=10000, sleep_s=1.2):
    params = {
        "output": "json",
        "countrycode": "ES",
        "boundingbox": f"({bbox['north']},{bbox['west']}),({bbox['south']},{bbox['east']})",
        "maxresults": str(maxresults),
        "compact": "true",
        "verbose": "false",
        "key": api_key,
    }
    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
    if r.status_code == 429:
        time.sleep(5)
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r.json()

# ----------------- Normalización filas + agregados -----------------
def normalize_rows(pois_json, ccaa_name: str, month_label: str, treat_free_as_zero=False):
    """
    Convierte JSON en filas: una por conexión donde tengamos €/kWh y banda.
    Guarda además un CSV QC con motivos de descarte.
    """
    rows = []
    qc_rows = []  
    for poi in pois_json:
        usage_cost = poi.get("UsageCost")
        lat = poi.get("AddressInfo", {}).get("Latitude")
        lon = poi.get("AddressInfo", {}).get("Longitude")
        operator = (poi.get("OperatorInfo") or {}).get("Title")
        connections = poi.get("Connections") or []
        for c in connections:
            power = c.get("PowerKW")
            level_id = c.get("LevelID")
            band = band_from_power_kw(power, level_id)
            val, why = usagecost_to_eur_per_kwh(usage_cost, treat_free_as_zero=treat_free_as_zero)

            qc_rows.append({
                "month": month_label, "ccaa": ccaa_name, "operator": operator,
                "power_kW": power, "band": band, "usagecost_original": usage_cost,
                "parsed": val, "reason": why
            })

            if (val is not None) and (band is not None):
                rows.append({
                    "month": month_label, "ccaa": ccaa_name, "band": band,
                    "eur_kWh": val, "lat": lat, "lon": lon,
                    "operator": operator, "power_kW": power
                })

    qc_path = Path(f"data/qc/qc_{ccaa_name.replace(' ', '_')}_{month_label}.csv")
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8")
    return rows

def aggregate_month(df: pd.DataFrame):
    """
    Agrega por month × ccaa × band → mediana, media y n.
    """
    if df.empty:
        return pd.DataFrame(columns=["month","ccaa","band","median_eur_kWh","mean_eur_kWh","n"])
    agg = (
        df.groupby(["month","ccaa","band"])["eur_kWh"]
          .agg(median="median", mean="mean", n="count")
          .reset_index()
          .rename(columns={"median": "median_eur_kWh", "mean": "mean_eur_kWh"})
    )
    return agg

def month_range_for_year(year: int):
    now = datetime.utcnow()
    end_month = 12 if year < now.year else now.month
    return [f"{year}-{m:02d}" for m in range(1, end_month+1)]

def process_one_month(month_label: str, only_ccaa, api_key: str, maxresults: int, overwrite: bool, treat_free_as_zero=False):
    ensure_dirs()
    HEADERS.update({"X-API-Key": api_key})

    ccaa_items = CCAA_BOUNDING_BOXES.items()
    if only_ccaa:
        ccaa_items = [(k, v) for k, v in CCAA_BOUNDING_BOXES.items() if k in set(only_ccaa)]

    all_rows = []
    for ccaa_name, bbox in ccaa_items:
        raw_path = Path(f"data/raw/ocm_{ccaa_name.replace(' ', '_')}_{month_label}.json")
        if raw_path.exists() and not overwrite:
            print(f"[SKIP] {ccaa_name} {month_label}: raw ya existe, leyendo…")
            pois = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            print(f"[INFO] Descargando {ccaa_name} {month_label} …")
            pois = fetch_ccaa(bbox, api_key=api_key, maxresults=maxresults)
            with raw_path.open("w", encoding="utf-8") as f:
                json.dump(pois, f, ensure_ascii=False, indent=2)

        rows = normalize_rows(pois, ccaa_name, month_label, treat_free_as_zero=treat_free_as_zero)
        print(f"[INFO] {ccaa_name} {month_label}: filas con precio parseado = {len(rows)}")
        all_rows.extend(rows)

    df_rows = pd.DataFrame(all_rows)
    df_agg = aggregate_month(df_rows)

    rows_path = Path(f"data/processed/ocm_rows_{month_label}.csv")
    agg_path = Path(f"data/processed/ocm_agg_{month_label}.csv")
    df_rows.to_csv(rows_path, index=False, encoding="utf-8")
    df_agg.to_csv(agg_path, index=False, encoding="utf-8")
    print(f"[OK] Mes {month_label} -> rows: {rows_path.name}, agg: {agg_path.name}")
    return df_rows, df_agg

def combine_year(year: int):
    ensure_dirs()
    files = sorted(Path("data/processed").glob(f"ocm_agg_{year}-*.csv"))
    if not files:
        print(f"[WARN] No hay agregados mensuales para {year}")
        return None
    parts = [pd.read_csv(fp) for fp in files]
    annual = (
        pd.concat(parts, ignore_index=True)
          .sort_values(["month","ccaa","band"])
          .reset_index(drop=True)
    )
    out = Path(f"data/processed/ocm_agg_{year}.csv")
    annual.to_csv(out, index=False, encoding="utf-8")
    print(f"[OK] Agregado anual -> {out.name}")
    return annual

def main():
    parser = argparse.ArgumentParser(description="Ingesta OCM por CCAA y agregado mensual/anual")
    parser.add_argument("--month", type=str, help="YYYY-MM. Si se da, procesa solo ese mes.")
    parser.add_argument("--year", type=int, help="YYYY. Si se da, procesa todos los meses del año.")
    parser.add_argument("--only", type=str, nargs="*", help="Lista de CCAA (como en config.py)")
    parser.add_argument("--overwrite", action="store_true", help="Forzar descarga si ya existe el raw del mes.")
    parser.add_argument("--maxresults", type=int, default=10000, help="Máximo de POIs por petición.")
    parser.add_argument("--free0", action="store_true", help="Tratar 'free/gratis' como 0.0")
    args = parser.parse_args()

    api_key = load_api_key()

    if args.month and args.year:
        raise SystemExit("Usa --month O --year, pero no ambos.")

    if args.month:
        process_one_month(args.month, args.only, api_key, args.maxresults, args.overwrite, treat_free_as_zero=args.free0)
        return

    if args.year:
        for m in month_range_for_year(args.year):
            print(f"\n===== Procesando {m} =====")
            process_one_month(m, args.only, api_key, args.maxresults, args.overwrite, treat_free_as_zero=args.free0)
        combine_year(args.year)
        return

    month_label = datetime.utcnow().strftime("%Y-%m")
    process_one_month(month_label, args.only, api_key, args.maxresults, args.overwrite, treat_free_as_zero=args.free0)

if __name__ == "__main__":
    main()
