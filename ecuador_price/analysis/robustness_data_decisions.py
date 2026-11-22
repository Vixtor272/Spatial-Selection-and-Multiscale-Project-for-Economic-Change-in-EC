#!/usr/bin/env python3
"""
robustness_data_decisions.py
============================
Comprobaciones sobre las decisiones de preparación de datos pedidas en las
revisiones de los Entregables 1 y 2. Reutiliza, sin modificarlo, el controlador
(ecuador_price/decomposition.py) y el motor de Kemp et al.; lo único que
cambia entre variantes es la tabla mensual parroquia x mes que entra al
controlador.

Variantes (vía de población, ventanas anuales):
  base      la especificación del texto: número de empresas = stock de
            sociedades privadas activas en el registro del SRI, en las celdas
            (parroquia x sección CIIU x mes) con al menos una declaración de
            ventas; ventas reales anuales > SALES_FLOOR en ambos extremos.
  meses_11  la base restringida a los once meses presentes en 2024
            (enero-octubre y diciembre), para los pasos 2022->2023 (para
            compararlo con el mismo paso a doce meses), 2023->2024 y el tramo
            2017->2024 con los mismos meses en ambos extremos.
  ventas_internas
            la base con las ventas SIN exportaciones (V_VTX - V_X) en lugar de
            las ventas totales (V_VTX). Responde a la pregunta de si el
            resultado depende de contar las exportaciones como ventas.
  registro_completo
            la base con OTRA definición del número de empresas: todas las
            sociedades privadas con RUC activo de la parroquia y el mes,
            incluidas las de celdas (parroquia x sección x mes) en las que
            ninguna sociedad declaró ventas. Las ventas no cambian. NO es una
            prueba de robustez: es otro objeto de estudio (el catastro completo
            de sociedades activas), que responde a dónde están las sociedades
            activas y no a dónde está la actividad declarada. Se calcula solo
            para delimitar el alcance de los resultados del trabajo.
            (El nombre de la variante se conserva; la fuente es la
            estadística de contribuyentes ACTIVOS del SRI, no el total de
            contribuyentes inscritos.)

Además escribe celdas_sin_declaracion.csv, que explica la diferencia entre la
base y esa comparación: qué parte de las sociedades activas está en celdas sin
declaración, y cuánto creció el stock de 2019 a 2020 con cada definición,
por tamaño de parroquia.

Salida: results/robustness_data_decisions/summary.csv
        (una fila por variante x ventana x vía),
        results/robustness_data_decisions/celdas_sin_declaracion.csv
        y un resumen en pantalla.

Ejecutar:
    python -m ecuador_price.analysis.robustness_data_decisions
"""
import pandas as pd

from ecuador_price import decomposition as dec
from ecuador_price import paths

FREQ = "yearly"
OUT = paths.ROBUSTNESS

MESES_2024 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]      # noviembre de 2024 no está en el extracto
PASOS = [(str(y), str(y + 1)) for y in range(2017, 2023)] + [("2017", "2023")]

CELDA = ["anio", "mes", "cod_ciuu4", "cod_parr"]       # parroquia x sección CIIU x mes
# tamaño de la parroquia: sociedades activas en el registro en 2019 (promedio mensual)
CLASES = [0, 10, 50, 500, float("inf")]
NOMBRES_CLASES = ["menos de 10", "10 a 50", "50 a 500", "más de 500"]


def tabla_mensual_base():
    """parroquia x año x mes con firms (registro, celdas declaradas) y sales reales."""
    return dec.monthly_table(dec.load_real_panel())


def tabla_mensual_ventas_internas():
    """La misma tabla, con ventas internas: total_sales (V_VTX) menos exports (V_X).
    El deflactor es el mismo. Las celdas que solo exportan quedan con ventas 0 y,
    si eso deja a una parroquia sin ventas en el año, la regla de validez la excluye."""
    panel = pd.read_parquet(paths.FIRM_REVENUE_PARQUET,
                            columns=["anio", "mes", "cod_parr", "total_sales",
                                     "exports", "total_firms"])
    panel["year"] = panel["anio"].astype(int)
    ipc = pd.read_csv(paths.IPC_CSV)
    ipc.columns = ["year", "factor"]
    factor = dict(zip(ipc["year"], ipc["factor"]))
    panel["real_sales"] = (panel["total_sales"] - panel["exports"]) * panel["year"].map(factor)
    return dec.monthly_table(panel)


def registro_por_celda():
    """El registro de sociedades privadas activas, una fila por celda, con una
    columna que dice si en esa celda alguna sociedad declaró ventas (es decir,
    si la celda está en la tabla de análisis)."""
    registro = pd.read_parquet(paths.ACTIVE_FIRMS_PARQUET)
    registro = (registro[registro["tipo"] == "SDD_NO_PUB"]
                .groupby(CELDA, as_index=False)["total"].sum())
    declaradas = pd.read_parquet(paths.FIRM_REVENUE_PARQUET, columns=CELDA).drop_duplicates()
    registro = registro.merge(declaradas.assign(con_declaracion=True), on=CELDA, how="left")
    registro["con_declaracion"] = registro["con_declaracion"].fillna(False).astype(bool)
    registro["year"] = registro["anio"].astype(int)
    registro["mes_i"] = registro["mes"].astype(int)
    return registro


def tabla_mensual_registro_completo(base, registro):
    """La tabla base con el número de empresas reemplazado por TODO el stock del
    registro en la parroquia y el mes. Las ventas y las filas son las de la base."""
    todo = (registro.groupby(["cod_parr", "year", "mes_i"], as_index=False)["total"].sum()
                    .rename(columns={"total": "firms"}))
    return base.drop(columns="firms").merge(todo, on=["cod_parr", "year", "mes_i"], how="left")


def celdas_sin_declaracion(base, completo, registro):
    """Por tamaño de parroquia: qué parte del registro está en celdas sin
    declaración (2019 y 2020) y cuánto creció el stock de 2019 a 2020 con cada
    definición del número de empresas."""
    stock_2019 = (registro[registro["year"] == 2019]
                  .groupby(["cod_parr", "mes_i"])["total"].sum().groupby("cod_parr").mean())
    clase = pd.cut(stock_2019, CLASES, labels=NOMBRES_CLASES)

    registro = registro.assign(clase=registro["cod_parr"].map(clase))
    sin = registro[~registro["con_declaracion"]]
    filas = pd.DataFrame({"parroquias": clase.value_counts()})
    for year in (2019, 2020):
        todo_año = registro[registro["year"] == year].groupby("clase", observed=True)["total"].sum()
        sin_año = sin[sin["year"] == year].groupby("clase", observed=True)["total"].sum()
        filas[f"sin_declaracion_{year}"] = sin_año / todo_año
    for nombre, tabla in [("base", base), ("registro_completo", completo)]:
        anual = (tabla[tabla["year"].isin([2019, 2020])]
                 .groupby(["cod_parr", "year"])["firms"].mean().unstack().dropna())
        por_clase = anual.groupby(anual.index.map(clase), observed=True).sum()
        filas[f"empresas_2019_{nombre}"] = por_clase[2019]
        filas[f"crecimiento_2019_2020_{nombre}"] = por_clase[2020] / por_clase[2019] - 1
    return filas.loc[NOMBRES_CLASES].rename_axis("sociedades_en_el_registro_2019").reset_index()


def correr(variante, monthly, pasos):
    pp = dec.parish_period_table(monthly, FREQ)
    valid = dec.annual_valid_cells(monthly)
    filas = []
    for k0, k1 in pasos:
        p0, p1 = (int(k0), 1), (int(k1), 1)
        nombre = f"{k0}_to_{k1}"
        res = dec.run_window(pp, valid, nombre, p0, p1, FREQ)   # sin CSV por unidad
        v0 = dec.valid_parishes(pp, k0, valid)
        v1 = dec.valid_parishes(pp, k1, valid)
        sobrev = v0 & v1
        n0 = pp[(pp.period == k0) & pp.cod_parr.isin(sobrev)]["firms"].sum()
        n1 = pp[(pp.period == k1) & pp.cod_parr.isin(sobrev)]["firms"].sum()
        for r in dec.national_summary_rows(res, nombre, p0, p1, "step", FREQ):
            r.update(variante=variante, parroquias=len(sobrev), empresas_0=n0, empresas_1=n1)
            filas.append(r)
    return filas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = tabla_mensual_base()
    filas = []

    print("\n### base")
    filas += correr("base", base, PASOS)

    print("\n### meses_11")
    once = base[base["mes_i"].isin(MESES_2024)]
    filas += correr("meses_11", once, [("2022", "2023"), ("2023", "2024"), ("2017", "2024")])

    print("\n### ventas_internas")
    filas += correr("ventas_internas", tabla_mensual_ventas_internas(), PASOS)

    print("\n### registro_completo")
    registro = registro_por_celda()
    completo = tabla_mensual_registro_completo(base, registro)
    filas += correr("registro_completo", completo, PASOS)
    celdas = celdas_sin_declaracion(base, completo, registro)
    celdas.to_csv(OUT / "celdas_sin_declaracion.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(filas)
    df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    ver = ["variante", "window", "parroquias", "empresas_0", "empresas_1", "delta_trait_per_year",
           "selection_total", "sel_between_provinces", "sel_canton_within_province",
           "sel_parish_within_canton", "transmission"]
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print("\n=== vía de población, por variante y ventana ===")
        print(df[df.path == "pop"][ver].to_string(index=False))
        print("\n=== registro en celdas sin declaración y crecimiento 2019 -> 2020, por tamaño de parroquia ===")
        print(celdas.to_string(index=False))
    print(f"\nguardado -> {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
