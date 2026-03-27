import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuração ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — tons de cinza claro / quase branco ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #f5f5f5;
        color: #1c1c1c;
    }

    .main { background-color: #f5f5f5; }

    /* Header */
    .header-box {
        background-color: #ffffff;
        border: 1px solid #e4e4e4;
        border-top: 3px solid #2c2c2c;
        padding: 22px 28px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .header-box h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        color: #1c1c1c;
        letter-spacing: 0.3px;
    }
    .header-box p {
        font-size: 0.82rem;
        color: #888;
        margin: 4px 0 0 0;
        font-weight: 400;
    }

    /* Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e4e4e4;
        border-radius: 4px;
        padding: 20px 22px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #999;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1c1c1c;
    }
    .metric-value.alto  { color: #b91c1c; }
    .metric-value.baixo { color: #166534; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 3px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .badge-alto  { background: #fff1f0; color: #b91c1c; border: 1px solid #fca5a5; }
    .badge-baixo { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }

    /* Título de seção */
    .section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #888;
        border-bottom: 1px solid #e4e4e4;
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }

    /* Tabela de perfil */
    .profile-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .profile-table tr { border-bottom: 1px solid #f0f0f0; }
    .profile-table td { padding: 7px 4px; }
    .profile-table td:first-child { color: #888; font-size: 0.8rem; }
    .profile-table td:last-child  { font-weight: 600; color: #1c1c1c; text-align: right; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e4e4e4;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #444 !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #1c1c1c !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    /* Box de seção */
    .section-box {
        background: #ffffff;
        border: 1px solid #e4e4e4;
        border-radius: 4px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Modelos ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    return model, scaler, imputer

model, scaler, imputer = load_models()

FEATURES = [
    'utilizacao_credito_rotativo', 'idade', 'atrasos_30_59_dias',
    'taxa_endividamento', 'renda_mensal', 'qtd_linhas_credito',
    'atrasos_90_dias', 'qtd_emprestimos_imoveis',
    'atrasos_60_89_dias', 'qtd_dependentes'
]

FEATURES_LABELS = {
    'utilizacao_credito_rotativo': 'Utilização Crédito Rotativo',
    'idade':                       'Idade',
    'atrasos_30_59_dias':          'Atrasos 30-59 dias',
    'taxa_endividamento':          'Taxa de Endividamento',
    'renda_mensal':                'Renda Mensal',
    'qtd_linhas_credito':          'Linhas de Crédito',
    'atrasos_90_dias':             'Atrasos 90+ dias',
    'qtd_emprestimos_imoveis':     'Empréstimos Imobiliários',
    'atrasos_60_89_dias':          'Atrasos 60-89 dias',
    'qtd_dependentes':             'Dependentes'
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>Credit Risk Prediction</h1>
    <p>Modelo preditivo de inadimplência &nbsp;&middot;&nbsp; XGBoost + Machine Learning &nbsp;&middot;&nbsp; Rafael Reghine Munhoz</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Dados do Cliente")
st.sidebar.markdown("---")

idade                   = st.sidebar.slider("Idade", 18, 100, 35)
renda_mensal            = st.sidebar.number_input("Renda Mensal (R$)", 0, 100000, 5000, step=500)
taxa_endividamento      = st.sidebar.slider("Taxa de Endividamento", 0.0, 5.0, 0.3, step=0.01)
utilizacao_credito      = st.sidebar.slider("Utilização do Crédito Rotativo", 0.0, 1.0, 0.3, step=0.01)
qtd_linhas_credito      = st.sidebar.slider("Linhas de Crédito Abertas", 0, 50, 5)
qtd_emprestimos_imoveis = st.sidebar.slider("Empréstimos Imobiliários", 0, 20, 1)
qtd_dependentes         = st.sidebar.slider("Dependentes", 0, 10, 0)
atrasos_30_59           = st.sidebar.slider("Atrasos 30-59 dias", 0, 15, 0)
atrasos_60_89           = st.sidebar.slider("Atrasos 60-89 dias", 0, 15, 0)
atrasos_90              = st.sidebar.slider("Atrasos 90+ dias", 0, 15, 0)

# ── Predição ──────────────────────────────────────────────────────────────────
input_data = pd.DataFrame([[
    utilizacao_credito, idade, atrasos_30_59,
    taxa_endividamento, renda_mensal, qtd_linhas_credito,
    atrasos_90, qtd_emprestimos_imoveis, atrasos_60_89, qtd_dependentes
]], columns=FEATURES)

input_imputed = imputer.transform(input_data)
input_scaled  = scaler.transform(input_imputed)
prob          = model.predict_proba(input_scaled)[0][1]
score         = int((1 - prob) * 1000)
classificacao = "ALTO RISCO" if prob >= 0.5 else "BAIXO RISCO"
cor_class     = "alto"       if prob >= 0.5 else "baixo"
badge_class   = "badge-alto" if prob >= 0.5 else "badge-baixo"

# ── Métricas ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Probabilidade de Inadimplência</div>
        <div class="metric-value {cor_class}">{prob:.1%}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Classificação de Risco</div>
        <div style="margin-top:6px">
            <span class="badge {badge_class}">{classificacao}</span>
        </div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Score de Crédito</div>
        <div class="metric-value">{score}<span style="font-size:1rem;color:#aaa"> /1000</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gráficos ──────────────────────────────────────────────────────────────────
col_esq, col_dir = st.columns([3, 2])

BG = '#ffffff'

with col_esq:
    st.markdown('<div class="section-title">Contribuição ao Risco por Variável</div>',
                unsafe_allow_html=True)

    pesos = {
        'atrasos_90_dias':              0.25,
        'utilizacao_credito_rotativo':  0.20,
        'atrasos_30_59_dias':           0.15,
        'atrasos_60_89_dias':           0.12,
        'taxa_endividamento':           0.10,
        'renda_mensal':                 0.08,
        'idade':                        0.04,
        'qtd_linhas_credito':           0.03,
        'qtd_emprestimos_imoveis':      0.02,
        'qtd_dependentes':              0.01,
    }
    valores_norm = {
        'atrasos_90_dias':              min(atrasos_90 / 5, 1.0),
        'utilizacao_credito_rotativo':  min(utilizacao_credito, 1.0),
        'atrasos_30_59_dias':           min(atrasos_30_59 / 5, 1.0),
        'atrasos_60_89_dias':           min(atrasos_60_89 / 5, 1.0),
        'taxa_endividamento':           min(taxa_endividamento / 5, 1.0),
        'renda_mensal':                 max(1 - renda_mensal / 20000, 0),
        'idade':                        max(1 - (idade - 18) / 60, 0),
        'qtd_linhas_credito':           min(qtd_linhas_credito / 20, 1.0),
        'qtd_emprestimos_imoveis':      min(qtd_emprestimos_imoveis / 10, 1.0),
        'qtd_dependentes':              min(qtd_dependentes / 5, 1.0),
    }
    scores_var = {k: valores_norm[k] * pesos[k] * 10 for k in pesos}
    df_sc = pd.DataFrame({
        'Variável': [FEATURES_LABELS[k] for k in scores_var],
        'Score':    list(scores_var.values())
    }).sort_values('Score', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cores = ['#b91c1c' if v >= 1.5 else '#d97706' if v >= 0.8 else '#16a34a'
             for v in df_sc['Score']]

    bars = ax.barh(df_sc['Variável'], df_sc['Score'],
                   color=cores, edgecolor='white', linewidth=0.8, height=0.55)

    for bar, val in zip(bars, df_sc['Score']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=8.5,
                color='#444', fontweight='600')

    ax.set_xlabel('Índice de Contribuição ao Risco', fontsize=8.5, color='#888')
    ax.set_xlim(0, df_sc['Score'].max() * 1.3 if df_sc['Score'].max() > 0 else 1)
    ax.tick_params(labelsize=8.5, colors='#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e4e4e4')
    ax.spines['bottom'].set_color('#e4e4e4')

    legenda = [
        mpatches.Patch(color='#b91c1c', label='Alto'),
        mpatches.Patch(color='#d97706', label='Médio'),
        mpatches.Patch(color='#16a34a', label='Baixo'),
    ]
    ax.legend(handles=legenda, loc='lower right', fontsize=8,
              framealpha=1, edgecolor='#e4e4e4', title='Nível',
              title_fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_dir:
    st.markdown('<div class="section-title">Perfil do Cliente</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <table class="profile-table">
        <tr><td>Idade</td>                   <td>{idade} anos</td></tr>
        <tr><td>Renda Mensal</td>            <td>R$ {renda_mensal:,.0f}</td></tr>
        <tr><td>Taxa de Endividamento</td>   <td>{taxa_endividamento:.2f}</td></tr>
        <tr><td>Utilização do Crédito</td>   <td>{utilizacao_credito:.0%}</td></tr>
        <tr><td>Linhas de Crédito</td>       <td>{qtd_linhas_credito}</td></tr>
        <tr><td>Empréstimos Imob.</td>       <td>{qtd_emprestimos_imoveis}</td></tr>
        <tr><td>Dependentes</td>             <td>{qtd_dependentes}</td></tr>
        <tr><td>Atrasos 30-59 dias</td>      <td>{atrasos_30_59}x</td></tr>
        <tr><td>Atrasos 60-89 dias</td>      <td>{atrasos_60_89}x</td></tr>
        <tr><td>Atrasos 90+ dias</td>        <td>{atrasos_90}x</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Probabilidade de Default</div>',
                unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4, 1.4))
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(BG)

    ax2.barh(0, 1.0, color='#f0f0f0', height=0.45)
    cor_g = '#b91c1c' if prob >= 0.5 else '#d97706' if prob >= 0.3 else '#16a34a'
    ax2.barh(0, prob, color=cor_g, height=0.45)
    ax2.axvline(prob, color='#1c1c1c', linewidth=1.8)
    ax2.text(prob, 0.32, f'{prob:.1%}', ha='center', va='bottom',
             fontsize=11, fontweight='700', color='#1c1c1c')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.4, 0.55)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(['0%', '50%', '100%'], fontsize=8, color='#888')
    ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── Mapa de sensibilidade ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Mapa de Sensibilidade — Atrasos 90+ dias vs Utilização do Crédito</div>',
            unsafe_allow_html=True)
st.markdown("""<p style="font-size:0.8rem;color:#888;margin-bottom:12px">
Simulação do risco conforme os dois principais preditores variam.
O marcador indica a posição atual do cliente.</p>""",
            unsafe_allow_html=True)

atrasos_range = np.arange(0, 8)
util_range    = np.linspace(0, 1, 8)
matriz        = np.zeros((len(util_range), len(atrasos_range)))

for i, u in enumerate(util_range):
    for j, a in enumerate(atrasos_range):
        inp = pd.DataFrame([[
            u, idade, atrasos_30_59, taxa_endividamento,
            renda_mensal, qtd_linhas_credito,
            a, qtd_emprestimos_imoveis, atrasos_60_89, qtd_dependentes
        ]], columns=FEATURES)
        inp_sc = scaler.transform(imputer.transform(inp))
        matriz[i, j] = model.predict_proba(inp_sc)[0][1]

fig3, ax3 = plt.subplots(figsize=(11, 4))
fig3.patch.set_facecolor(BG)
ax3.set_facecolor(BG)

im = ax3.imshow(matriz, aspect='auto', cmap='RdYlGn_r',
                vmin=0, vmax=1, origin='lower')

ax3.set_xticks(range(len(atrasos_range)))
ax3.set_xticklabels([str(int(a)) for a in atrasos_range], fontsize=9, color='#444')
ax3.set_yticks(range(len(util_range)))
ax3.set_yticklabels([f'{u:.0%}' for u in util_range], fontsize=9, color='#444')
ax3.set_xlabel('Atrasos 90+ dias', fontsize=9, color='#888')
ax3.set_ylabel('Utilização do Crédito Rotativo', fontsize=9, color='#888')

for i in range(len(util_range)):
    for j in range(len(atrasos_range)):
        val = matriz[i, j]
        cor_txt = 'white' if val > 0.55 else '#1c1c1c'
        ax3.text(j, i, f'{val:.0%}', ha='center', va='center',
                 fontsize=8, color=cor_txt, fontweight='600')

x_atual = min(atrasos_90, 7)
y_atual = np.argmin(np.abs(util_range - utilizacao_credito))
ax3.plot(x_atual, y_atual, 'o', markersize=16, color='white',
         markeredgecolor='#1c1c1c', markeredgewidth=2.5)
ax3.text(x_atual, y_atual, 'C', ha='center', va='center',
         fontsize=7.5, fontweight='900', color='#1c1c1c')

cbar = plt.colorbar(im, ax=ax3, shrink=0.9)
cbar.set_label('Probabilidade de Inadimplência', fontsize=8, color='#888')
cbar.ax.tick_params(labelsize=8, colors='#444')

for s in ax3.spines.values():
    s.set_color('#e4e4e4')

plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="font-size:0.76rem;color:#bbb;text-align:center">
    Credit Risk Prediction &nbsp;&middot;&nbsp; Rafael Reghine Munhoz &nbsp;&middot;&nbsp;
    <a href="https://github.com/reghine" style="color:#aaa">GitHub</a> &nbsp;&middot;&nbsp;
    <a href="https://linkedin.com/in/rafaelreghine" style="color:#aaa">LinkedIn</a>
</p>
""", unsafe_allow_html=True)
