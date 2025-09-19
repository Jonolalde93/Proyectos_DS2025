import argparse
from pathlib import Path
import pandas as pd

def load_monthlies_for_year(year: int):
    files = sorted(Path("data/processed").glob(f"ocm_agg_{year}-*.csv"))
    if not files:
        raise SystemExit(f"No se encontraron agregados mensuales para {year} en data/processed/")
    parts = [pd.read_csv(fp) for fp in files]
    df = pd.concat(parts, ignore_index=True)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values(["month","ccaa","band"]).reset_index(drop=True)
    return df

def make_wide(df: pd.DataFrame, value_col: str):
    """
    value_col: 'median_eur_kWh' o 'mean_eur_kWh'
    Devuelve tabla wide: month, ccaa, AC, DC
    """
    wide = (
        df.pivot_table(index=["month","ccaa"], columns="band", values=value_col)
          .reset_index()
          .sort_values(["month","ccaa"])
    )
    wide["month"] = wide["month"].dt.strftime("%Y-%m")
    return wide

def main():
    p = argparse.ArgumentParser(description="Genera CSVs 'wide' con media/mediana por mes")
    p.add_argument("--year", type=int, required=True, help="Año a procesar (YYYY)")
    args = p.parse_args()

    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_monthlies_for_year(args.year)

    wide_median = make_wide(df, "median_eur_kWh")
    wide_mean   = make_wide(df, "mean_eur_kWh")

    f1 = outdir / f"ocm_curvas_median_{args.year}_wide.csv"
    f2 = outdir / f"ocm_curvas_mean_{args.year}_wide.csv"

    wide_median.to_csv(f1, index=False, encoding="utf-8")
    wide_mean.to_csv(f2, index=False, encoding="utf-8")

    print(f"[OK] Guardado: {f1.name}")
    print(f"[OK] Guardado: {f2.name}")

if __name__ == "__main__":
    main()
