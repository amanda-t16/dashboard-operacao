import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# =========================
# 1) CONFIG DB
# =========================
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_HOST = st.secrets["DB_HOST"]
DB_PORT = st.secrets["DB_PORT"]
DB_NAME = st.secrets["DB_NAME"]

PROJECT_TABLES = ["neo", "enel", "cpfl_cloud", "CPFL"]  # nomes EXATOS das tabelas

# Expressões normalizadas (pra não vir None)
CAMPANHA_EXPR = "COALESCE(NULLIF(TRIM(campanha), ''), 'SEM_CAMPANHA')"
CIDADE_EXPR   = "COALESCE(NULLIF(TRIM(cidade), ''), 'SEM_CIDADE')"

# =========================
# ✅ REGRA: STATUS QUE CONTAM COMO "ABORDADOS"
# =========================
STATUS_ABORDADOS = [
    "Agendamento Pessoal – Com Cliente",
    "Agendamento Base – Com Cliente",
    "Ligação transferida pelo Localizador",
    "Ligação transferida pelo Voicebot",
    "Pré-venda",
    "Venda confirmada",
    "Manter pendente",
    "Transferência para a auditoria – pendência do operador",
    "Agente Virtual – Recusou falar",
    "Agente Virtual – Cliente já pagou",
    "Agente Virtual – Recusou negociar",
    "Agente Virtual – Webservice Erro",
    "Agente Virtual – Não possui cobrança pendente",
    "Agente Virtual – Acordo Efetivado",
    "Agente Virtual – Cliente desligou – questionamento já pagou a fatura",
    "Agente Virtual – Cliente desligou – questionamento pagamento - 1 parcela",
    "Agente Virtual – Cliente desligou – questionamento pagamento - 3 parcelas",
    "Agente Virtual – Cliente desligou – questionamento pagamento - 6 parcelas",
    "Agente Virtual – Falecido",
    "Agente Virtual – SMS enviado",
    "Drop do cliente através do botão da página",
    "Drop automático – browser finalizado",
    "Drop automático – CRM finalizado",
    "Drop automático – expirado em fila",
    "Chat finalizado pelo cliente",
    "Finalização automática (sem interação)",
    "Status recusa padrão",
    "Cliente desligou",
    "Cliente ríspido",
    "Caiu ligação – Em atendimento",
    "Já possui outro seguro",
    "Não possui UC",
    "Não quer ser contatado",
    "Não tem interesse – inclui familiar",
    "Recusou confirmar dados",
    "Pendente de análise",
    "Pré-Auditoria",
    "Contenção de despesas / desempregado",
    "Já foi contatado pela Assistencial",
    "CQ – Não fez sondagens (titular / responsável financeiro)",
    "CQ – Não desvincula a Assisty 24h da empresa de energia",
    "CQ – Cliente não entendeu que se trata de uma adesão",
    "CQ – Não refaz auditoria quando necessário",
    "CQ – Não solicitou a nova Unidade Consumidora / Código do cliente",
    "CQ – Não informou as fraseologias obrigatórias ou não verbalizou na integra",
    "CQ – Passou informações e/ou coletou dados com terceiros",
    "CQ – Clareza nas informações (aceleração / baixa o tom de voz)",
    "CQ – Postura e Ética (rispidez / elevação de tom de voz / ironia)",
    "CQ – Não faz Mini Venda quando necessário",
    "CQ – Barulho externo / RC não sonda entendimento do cliente",
    "CQ – Insistência com o cliente",
    "CQ – Não deixa claro que não possui acesso aos dados",
    "CQ – Respondeu parcialmente às dúvidas do cliente",
    "CQ – RC não é objetivo ao responder o questionamento do cliente",
]

# =========================
# ✅ REGRAS FIXAS (SEM FILTROS GLOBAIS)
# =========================
VENDA_KEYWORDS = ["venda", "aprov", "contrat", "efetiv"]
CANCELAMENTO_KEYWORDS = ["cancel"]
DT_INI = None  # None = sem filtro por data
DT_FIM = None  # None = sem filtro por data

# =========================
# 2) ENGINE
# =========================
@st.cache_resource
def make_engine():
    url = URL.create(
        drivername="mysql+pymysql",
        username=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )
    engine = create_engine(
        url,
        future=True,
        connect_args={"charset": "utf8mb4"},
        pool_pre_ping=True
    )
    with engine.connect() as conn:
        conn.execute(text("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.commit()
    return engine

engine = make_engine()

# =========================
# 3) HELPERS
# =========================
def force_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return ((n * 100) / d).astype(float).round(2)

def _sql_in_list(values, prefix="v"):
    params = {}
    placeholders = []
    for i, v in enumerate(values):
        key = f"{prefix}{i}"
        params[key] = v
        placeholders.append(f":{key}")
    return ", ".join(placeholders), params

def build_where(campanhas, dt_ini, dt_fim):
    where = ["1=1"]
    params = {}

    if campanhas:
        in_sql, in_params = _sql_in_list(campanhas, prefix="camp_")
        where.append(f"{CAMPANHA_EXPR} IN ({in_sql})")
        params.update(in_params)

    if dt_ini:
        where.append("data_ultima_ligacao >= :dt_ini")
        params["dt_ini"] = datetime.combine(dt_ini, datetime.min.time())

    if dt_fim:
        dt_fim_next = datetime.combine(dt_fim, datetime.min.time()) + timedelta(days=1)
        where.append("data_ultima_ligacao < :dt_fim_next")
        params["dt_fim_next"] = dt_fim_next

    return " AND ".join(where), params

def like_any(field, keywords, prefix):
    if not keywords:
        return "0=1", {}
    parts, p = [], {}
    for i, kw in enumerate(keywords):
        key = f"{prefix}{i}"
        p[key] = f"%{kw.lower()}%"
        parts.append(f"LOWER(COALESCE({field}, '')) LIKE :{key}")
    return "(" + " OR ".join(parts) + ")", p

def in_or_false(field, values, prefix):
    if not values:
        return "0=1", {}
    in_sql, in_params = _sql_in_list(values, prefix=prefix)
    return f"{field} IN ({in_sql})", in_params

def infer_status_groups(status_list):
    low = [(s, str(s).lower()) for s in status_list]

    def pick(keys):
        out = []
        for orig, l in low:
            if any(k in l for k in keys):
                out.append(orig)
        return sorted(list(set(out)))

    return {
        "disponiveis": pick(["dispon", "nome dispon", "disponível", "disponivel"]),
        "virgens": pick(["virgem"]),
        "secretaria": pick(["secret", "eletr"]),
        "caixa_postal": pick(["caixa postal", "postal", "cx postal"]),
        "nao_atende": pick(["nao atende", "não atende"]),
        "qualquer_caixa": pick(["caixa postal", "postal", "cx postal"]),
    }

# ✅ TOTALIZADOR (linha TOTAL somando colunas numéricas)
def add_totals_row(df: pd.DataFrame, label_col: str, label_value="TOTAL"):
    if df is None or df.empty:
        return df

    total_row = {}
    for col in df.columns:
        if col == label_col:
            total_row[col] = label_value
        elif pd.api.types.is_numeric_dtype(df[col]):
            total_row[col] = pd.to_numeric(df[col], errors="coerce").sum()
        else:
            total_row[col] = ""

    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# =========================
# 4) UI
# =========================
st.set_page_config(page_title="Dashboard Operação", layout="wide")
st.title("📊 Dashboard - Desempenho bases")

# =========================
# 5) TABS
# =========================
tabs = st.tabs(PROJECT_TABLES)

for table_name, tab in zip(PROJECT_TABLES, tabs):
    with tab:
        st.subheader(f"Projeto: {table_name}")

        @st.cache_data(ttl=120)
        def distinct_vals(table, col_expr):
            q = text(f"""
                SELECT DISTINCT {col_expr} AS v
                FROM {table}
                WHERE {col_expr} IS NOT NULL AND TRIM({col_expr}) <> ''
                ORDER BY 1
            """)
            dfv = pd.read_sql(q, engine)
            return [str(x).strip() for x in dfv["v"].tolist() if x is not None and str(x).strip() != ""]

        campanhas_disp = distinct_vals(table_name, CAMPANHA_EXPR)
        status_disp = distinct_vals(table_name, "status_mailing")

        campanhas_sel = st.multiselect(
            "Filtrar por campanha (afeta TODOS os visuais abaixo)",
            campanhas_disp,
            default=[],
            key=f"{table_name}_campanhas"
        )

        # ✅ usa DT_INI / DT_FIM fixos (sem filtro por data)
        where_sql, base_params = build_where(campanhas_sel, DT_INI, DT_FIM)

        defaults = infer_status_groups(status_disp)
        st_disponiveis = defaults["disponiveis"]
        st_virgens = defaults["virgens"]
        st_secretaria = defaults["secretaria"]
        st_caixa = defaults["caixa_postal"]
        st_nao_atende = defaults["nao_atende"]
        st_qualquer_caixa = defaults["qualquer_caixa"]

        # ✅ usa keywords fixas (sem sidebar)
        venda_expr1, p_v1 = like_any("status_proposta", VENDA_KEYWORDS, f"{table_name}_vk1_")
        venda_expr2, p_v2 = like_any("situacao_proposta", VENDA_KEYWORDS, f"{table_name}_vk2_")
        canc_expr1, p_c1 = like_any("status_proposta", CANCELAMENTO_KEYWORDS, f"{table_name}_ck1_")
        canc_expr2, p_c2 = like_any("situacao_proposta", CANCELAMENTO_KEYWORDS, f"{table_name}_ck2_")

        disp_expr, p_disp = in_or_false("status_mailing", st_disponiveis, f"{table_name}_disp_")
        vir_expr, p_vir = in_or_false("status_mailing", st_virgens, f"{table_name}_vir_")
        sec_expr, p_sec = in_or_false("status_mailing", st_secretaria, f"{table_name}_sec_")
        cx_expr, p_cx = in_or_false("status_mailing", st_caixa, f"{table_name}_cx_")
        na_expr, p_na = in_or_false("status_mailing", st_nao_atende, f"{table_name}_na_")
        qc_expr, p_qc = in_or_false("status_mailing", st_qualquer_caixa, f"{table_name}_qc_")

        abord_expr, p_abord = in_or_false("status_mailing", STATUS_ABORDADOS, f"{table_name}_abord_")

        params_all = {}
        params_all.update(base_params)
        params_all.update(p_v1); params_all.update(p_v2)
        params_all.update(p_c1); params_all.update(p_c2)
        params_all.update(p_disp); params_all.update(p_vir); params_all.update(p_sec)
        params_all.update(p_cx); params_all.update(p_na); params_all.update(p_qc)
        params_all.update(p_abord)

        # =========================
        # VISUAL 1: STATUS POR CAMPANHA
        # =========================
        st.markdown("## 1) Contagem de Status (por campanha)")

        q_status = f"""
            SELECT
                {CAMPANHA_EXPR} AS campanha,
                COUNT(*) AS base_total,
                SUM(CASE WHEN ({abord_expr}) THEN 1 ELSE 0 END) AS abordados,
                SUM(CASE WHEN COALESCE(status_proposta,'') <> '' OR COALESCE(situacao_proposta,'') <> '' THEN 1 ELSE 0 END) AS propostas_cadastradas,

                SUM(CASE WHEN ({venda_expr1} OR {venda_expr2}) THEN 1 ELSE 0 END) AS vendas,
                SUM(CASE WHEN ({canc_expr1} OR {canc_expr2}) THEN 1 ELSE 0 END) AS canceladas,

                SUM(CASE WHEN ({disp_expr}) THEN 1 ELSE 0 END) AS status_disponiveis,
                SUM(CASE WHEN ({vir_expr}) THEN 1 ELSE 0 END) AS status_virgens,
                SUM(CASE WHEN ({sec_expr}) THEN 1 ELSE 0 END) AS status_secretaria_eletronica,
                SUM(CASE WHEN ({cx_expr}) THEN 1 ELSE 0 END) AS status_caixa_postal,
                SUM(CASE WHEN ({na_expr}) THEN 1 ELSE 0 END) AS status_nao_atende,
                SUM(CASE WHEN ({qc_expr}) THEN 1 ELSE 0 END) AS status_qualquer_caixa_postal
            FROM {table_name}
            WHERE {where_sql}
            GROUP BY {CAMPANHA_EXPR}
            ORDER BY base_total DESC;
        """

        df_status = pd.read_sql(text(q_status), engine, params=params_all)
        df_status = force_numeric(df_status, [
            "base_total", "abordados", "propostas_cadastradas", "vendas", "canceladas",
            "status_disponiveis", "status_virgens", "status_secretaria_eletronica",
            "status_caixa_postal", "status_nao_atende", "status_qualquer_caixa_postal"
        ])

        if df_status.empty:
            st.info("Sem dados para os filtros selecionados.")
        else:
            df_status["conv_por_abordados_%"] = safe_pct(df_status["vendas"], df_status["abordados"])
            df_status["conv_por_base_total_%"] = safe_pct(df_status["vendas"], df_status["base_total"])

            # ✅ adiciona TOTAL
            df_status_total = add_totals_row(df_status, label_col="campanha", label_value="TOTAL")
            st.dataframe(df_status_total, use_container_width=True)

            st.markdown("### Vendas por campanha")
            st.bar_chart(df_status[["campanha", "vendas"]].set_index("campanha"), use_container_width=True)

        # =========================
        # VISUAL 1.1: RESUMO POR DATA
        # =========================
        st.markdown("## 1.1) Resumo por Data (data_ultima_ligacao)")

        q_date = f"""
            SELECT
                DATE(data_ultima_ligacao) AS data,
                COUNT(*) AS registros,
                SUM(CASE WHEN ({abord_expr}) THEN 1 ELSE 0 END) AS abordados,
                SUM(CASE WHEN ({venda_expr1} OR {venda_expr2}) THEN 1 ELSE 0 END) AS vendas,
                SUM(CASE WHEN ({canc_expr1} OR {canc_expr2}) THEN 1 ELSE 0 END) AS canceladas
            FROM {table_name}
            WHERE {where_sql}
            GROUP BY DATE(data_ultima_ligacao)
            ORDER BY DATE(data_ultima_ligacao) DESC
            LIMIT 90;
        """

        df_date = pd.read_sql(text(q_date), engine, params=params_all)
        df_date = force_numeric(df_date, ["registros", "abordados", "vendas", "canceladas"])

        if df_date.empty:
            st.caption("Sem dados por data para os filtros atuais.")
        else:
            df_date["conv_por_abordados_%"] = safe_pct(df_date["vendas"], df_date["abordados"])

            # ✅ adiciona TOTAL
            df_date_total = add_totals_row(df_date, label_col="data", label_value="TOTAL")
            st.dataframe(df_date_total, use_container_width=True)

        # =========================
        # VISUAL 2: ANALÍTICA POR CIDADE
        # =========================
        st.markdown("## 2) Analítica por Cidade")

        q_city = f"""
            SELECT
                {CIDADE_EXPR} AS cidade,
                COUNT(DISTINCT NULLIF(TRIM(cpf_cnpj), '')) AS cpfs_unicos,
                COUNT(*) AS base_total,
                SUM(CASE WHEN ({abord_expr}) THEN 1 ELSE 0 END) AS abordados,
                SUM(CASE WHEN ({venda_expr1} OR {venda_expr2}) THEN 1 ELSE 0 END) AS vendas,
                SUM(CASE WHEN ({canc_expr1} OR {canc_expr2}) THEN 1 ELSE 0 END) AS canceladas
            FROM {table_name}
            WHERE {where_sql}
            GROUP BY {CIDADE_EXPR}
            ORDER BY vendas DESC
            LIMIT 300;
        """

        df_city = pd.read_sql(text(q_city), engine, params=params_all)
        df_city = force_numeric(df_city, ["cpfs_unicos", "base_total", "abordados", "vendas", "canceladas"])

        if df_city.empty:
            st.info("Sem dados por cidade para os filtros selecionados.")
        else:
            df_city["conv_%"] = safe_pct(df_city["vendas"], df_city["abordados"])
            df_city["cancel_%"] = safe_pct(df_city["canceladas"], df_city["vendas"])

            # ✅ adiciona TOTAL
            df_city_total = add_totals_row(df_city, label_col="cidade", label_value="TOTAL")
            st.dataframe(df_city_total, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Top cidades por vendas")
                st.bar_chart(df_city[["cidade", "vendas"]].set_index("cidade"), use_container_width=True)
            with c2:
                st.markdown("### Top cidades por conversão (%)")
                conv_plot = df_city[["cidade", "conv_%"]].dropna().set_index("cidade")
                st.bar_chart(conv_plot, use_container_width=True)

st.caption("✅ Dashboard somente consulta. Filtros removidos (somente campanha por aba).")