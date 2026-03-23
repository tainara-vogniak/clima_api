"""
Assistente de Previsão do Tempo — Backend FastAPI
Instalar: pip install fastapi uvicorn groq requests python-dotenv
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from datetime import datetime, timedelta
import unicodedata
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()

# ── Utilitário ────────────────────────────────────────────────────────────────
def unidecode(texto: str) -> str:
    """Remove acentos e caracteres especiais — substituto nativo do unidecode."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY_  = os.getenv("GROQ_API_KEY", "")
MODEL          = "llama-3.3-70b-versatile"
SLIDING_WINDOW = 10

sessoes: dict[str, dict] = {}

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache_previsao: dict = {}
_cache_geocode:  dict = {}
CACHE_TTL_MINUTOS = 15


# ── Mapa de siglas → nome completo do estado ──────────────────────────────────
ESTADOS_BR = {
    "AC": "Acre",             "AL": "Alagoas",               "AP": "Amapá",
    "AM": "Amazonas",         "BA": "Bahia",                 "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo",        "GO": "Goiás",
    "MA": "Maranhão",         "MT": "Mato Grosso",           "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",     "PA": "Pará",                  "PB": "Paraíba",
    "PR": "Paraná",           "PE": "Pernambuco",            "PI": "Piauí",
    "RJ": "Rio de Janeiro",   "RN": "Rio Grande do Norte",   "RS": "Rio Grande do Sul",
    "RO": "Rondônia",         "RR": "Roraima",               "SC": "Santa Catarina",
    "SP": "São Paulo",        "SE": "Sergipe",               "TO": "Tocantins",
}

# Mapa inverso: nome completo (sem acento, minúsculo) → sigla
ESTADOS_NOME_PARA_SIGLA = {
    unidecode(v).lower(): k for k, v in ESTADOS_BR.items()
}


# ── Normalização de entrada ───────────────────────────────────────────────────
def normalizar_entrada(cidade: str, estado: str) -> tuple[str, str]:
    cidade = cidade.strip()
    estado_limpo = estado.strip().upper()

    if estado_limpo in ESTADOS_BR:
        return cidade, estado_limpo

    estado_sem_acento = unidecode(estado.strip()).lower()
    if estado_sem_acento in ESTADOS_NOME_PARA_SIGLA:
        return cidade, ESTADOS_NOME_PARA_SIGLA[estado_sem_acento]

    separadores = re.split(r"[/,]", cidade)
    if len(separadores) == 2:
        parte_a = separadores[0].strip()
        parte_b = separadores[1].strip().upper()

        if parte_b in ESTADOS_BR:
            return parte_a, parte_b
        if parte_a.upper() in ESTADOS_BR:
            return parte_b, parte_a.upper()

        parte_b_sem_acento = unidecode(separadores[1].strip()).lower()
        if parte_b_sem_acento in ESTADOS_NOME_PARA_SIGLA:
            return parte_a, ESTADOS_NOME_PARA_SIGLA[parte_b_sem_acento]

    return cidade, estado_limpo


# ── Modelos de request ────────────────────────────────────────────────────────
class IniciarRequest(BaseModel):
    cidade: str
    estado: str

class ChatRequest(BaseModel):
    session_id: str
    mensagem: str


# ── Geocodificação ────────────────────────────────────────────────────────────
def _montar_geo(loc: dict, fallback: str) -> dict:
    return {
        "lat":  loc["latitude"],
        "lon":  loc["longitude"],
        "nome": loc.get("name", fallback),
        "pais": loc.get("country", "Brasil"),
    }

def geocode(cidade: str, estado: str) -> dict | None:
    chave = f"{unidecode(cidade).lower()}_{estado.lower()}"

    if chave in _cache_geocode:
        print(f"[geocode] Retornando cache para {chave}")
        return _cache_geocode[chave]

    url    = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": cidade, "count": 10, "language": "pt", "format": "json"}
    try:
        print(f"[geocode] Buscando: {cidade}, {estado}")
        r = requests.get(url, params=params, timeout=10)
        print(f"[geocode] Status HTTP: {r.status_code}")
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            print("[geocode] Nenhum resultado encontrado.")
            return None

        estado_upper = estado.strip().upper()
        for loc in results:
            if loc.get("country_code","").upper() == "BR" and loc.get("admin1_code","").upper() == estado_upper:
                geo = _montar_geo(loc, cidade)
                _cache_geocode[chave] = geo
                return geo
        for loc in results:
            if loc.get("country_code","").upper() == "BR":
                geo = _montar_geo(loc, cidade)
                _cache_geocode[chave] = geo
                return geo

        geo = _montar_geo(results[0], cidade)
        _cache_geocode[chave] = geo
        return geo
    except Exception as e:
        print(f"[Erro geocodificação] TIPO: {type(e).__name__} | MSG: {e}")
    return None


# ── Previsão do tempo ─────────────────────────────────────────────────────────
def obter_previsao(lat: float, lon: float) -> dict | None:
    chave = (round(lat, 3), round(lon, 3))
    agora = datetime.now()

    if chave in _cache_previsao:
        dados, timestamp = _cache_previsao[chave]
        if agora - timestamp < timedelta(minutes=CACHE_TTL_MINUTOS):
            print(f"[previsao] Retornando cache para {chave}")
            return dados

    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m","apparent_temperature","weathercode",
                    "windspeed_10m","precipitation","relative_humidity_2m"],
        "hourly": ["temperature_2m","precipitation_probability","weathercode"],
        "forecast_days": 1,
        "timezone": "America/Sao_Paulo",
    }
    try:
        print(f"[previsao] Buscando lat={lat}, lon={lon}")
        r = requests.get(url, params=params, timeout=10)
        print(f"[previsao] Status HTTP: {r.status_code}")
        r.raise_for_status()
        dados = r.json()
        _cache_previsao[chave] = (dados, agora)
        return dados
    except Exception as e:
        print(f"[Erro previsão] TIPO: {type(e).__name__} | MSG: {e}")
    return None

def codigo_para_descricao(wmo: int) -> str:
    tabela = {
        0:"Céu limpo",        1:"Predominantemente limpo", 2:"Parcialmente nublado",
        3:"Nublado",          45:"Neblina",                48:"Neblina com geada",
        51:"Garoa leve",      53:"Garoa moderada",         55:"Garoa intensa",
        61:"Chuva leve",      63:"Chuva moderada",         65:"Chuva intensa",
        71:"Neve leve",       73:"Neve moderada",          75:"Neve intensa",
        80:"Pancadas leves",  81:"Pancadas moderadas",     82:"Pancadas violentas",
        95:"Tempestade",      96:"Tempestade com granizo", 99:"Tempestade severa",
    }
    return tabela.get(wmo, f"Código {wmo}")

def formatar_dados_clima(geo: dict, dados: dict) -> str:
    cur        = dados.get("current", {})
    hora_atual = datetime.now().hour
    hourly     = dados.get("hourly", {})
    prec_prob  = hourly.get("precipitation_probability", [])
    temps      = hourly.get("temperature_2m", [])
    wcodes     = hourly.get("weathercode", [])
    proximas   = []
    for i in range(hora_atual, min(hora_atual + 6, len(prec_prob))):
        proximas.append(f"  {i:02d}h → {temps[i]:.1f}°C, chuva {prec_prob[i]}%, {codigo_para_descricao(wcodes[i])}")
    return f"""
=== DADOS METEOROLÓGICOS ===
Local    : {geo['nome']}, Brasil
Hora     : {datetime.now().strftime('%d/%m/%Y %H:%M')}
Condição : {codigo_para_descricao(cur.get('weathercode', 0))}
Temp.    : {cur.get('temperature_2m', 'N/D')}°C (sensação {cur.get('apparent_temperature', 'N/D')}°C)
Umidade  : {cur.get('relative_humidity_2m', 'N/D')}%
Vento    : {cur.get('windspeed_10m', 'N/D')} km/h
Precip.  : {cur.get('precipitation', 0)} mm
Próximas horas:
{chr(10).join(proximas) if proximas else '  (sem dados)'}
============================"""


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/iniciar")
def iniciar(req: IniciarRequest):
    cidade, estado = normalizar_entrada(req.cidade, req.estado)

    geo = geocode(cidade, estado)
    if not geo:
        raise HTTPException(status_code=404, detail=f"Cidade '{cidade}' não encontrada.")

    dados_clima = obter_previsao(geo["lat"], geo["lon"])
    if not dados_clima:
        raise HTTPException(status_code=502, detail="Erro ao obter dados de clima.")

    bloco_clima = formatar_dados_clima(geo, dados_clima)
    session_id  = f"{cidade.lower()}_{estado.lower()}_{datetime.now().timestamp()}"

    sessoes[session_id] = {
        "sistema": f"""Você é um assistente meteorológico inteligente e amigável para {geo['nome']}.
Use os dados abaixo como base para respostas sobre clima:

{bloco_clima}

Regras:
- Responda SEMPRE em português brasileiro.
- Seja conciso e claro; adicione dicas práticas (o que vestir, guarda-chuva, etc.).
- Emita ALERTAS em maiúsculas se houver risco (tempestade, calor extremo, frio intenso).
- Se perguntarem algo fora do clima, responda brevemente e redirecione ao tema.""",
        "historico":   [],
        "geo":         geo,
        "clima_atual": {
            "temperatura": dados_clima["current"].get("temperature_2m"),
            "sensacao":    dados_clima["current"].get("apparent_temperature"),
            "umidade":     dados_clima["current"].get("relative_humidity_2m"),
            "vento":       dados_clima["current"].get("windspeed_10m"),
            "precipitacao":dados_clima["current"].get("precipitation", 0),
            "condicao":    codigo_para_descricao(dados_clima["current"].get("weathercode", 0)),
        }
    }

    return {
        "session_id": session_id,
        "cidade":     geo["nome"],
        "estado":     ESTADOS_BR.get(estado, estado),
        "clima":      sessoes[session_id]["clima_atual"],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada. Reinicie.")

    sessao    = sessoes[req.session_id]
    historico = sessao["historico"]

    historico.append({"role": "user", "content": req.mensagem})
    janela = historico[-SLIDING_WINDOW:]

    client = Groq(api_key=GROQ_API_KEY_)
    try:
        resposta = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": sessao["sistema"]}] + janela,
            max_tokens=1024,
            temperature=0.7,
        )
        texto = resposta.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na API Groq: {e}")

    historico.append({"role": "assistant", "content": texto})
    return {"resposta": texto}
