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
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # carrega variáveis do arquivo .env

app = FastAPI()

# Permite requisições do frontend (necessário para CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve os arquivos estáticos do frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY","")
MODEL          = "llama-3.3-70b-versatile"
SLIDING_WINDOW = 10

# Armazena sessões de conversa em memória (por session_id)
sessoes: dict[str, list[dict]] = {}


# ── Modelos de request ────────────────────────────────────────────────────────
class IniciarRequest(BaseModel):
    cidade: str
    estado: str

class ChatRequest(BaseModel):
    session_id: str
    mensagem: str


# ── Geocodificação ────────────────────────────────────────────────────────────
def geocode(cidade: str, estado: str) -> dict | None:
    url    = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": cidade, "count": 10, "language": "pt", "format": "json"}
    try:
        r       = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        estado_upper = estado.strip().upper()
        for loc in results:
            if loc.get("country_code","").upper() == "BR" and loc.get("admin1_code","").upper() == estado_upper:
                return _montar_geo(loc, cidade)
        for loc in results:
            if loc.get("country_code","").upper() == "BR":
                return _montar_geo(loc, cidade)
        return _montar_geo(results[0], cidade)
    except Exception as e:
        print(f"[Erro geocodificação] {e}")
    return None

def _montar_geo(loc: dict, fallback: str) -> dict:
    return {
        "lat":  loc["latitude"],
        "lon":  loc["longitude"],
        "nome": loc.get("name", fallback),
        "pais": loc.get("country", "Brasil"),
    }


# ── Previsão do tempo ─────────────────────────────────────────────────────────
def obter_previsao(lat: float, lon: float) -> dict | None:
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
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Erro previsão] {e}")
    return None

def codigo_para_descricao(wmo: int) -> str:
    tabela = {
        0:"Céu limpo",1:"Predominantemente limpo",2:"Parcialmente nublado",
        3:"Nublado",45:"Neblina",48:"Neblina com geada",
        51:"Garoa leve",53:"Garoa moderada",55:"Garoa intensa",
        61:"Chuva leve",63:"Chuva moderada",65:"Chuva intensa",
        71:"Neve leve",73:"Neve moderada",75:"Neve intensa",
        80:"Pancadas leves",81:"Pancadas moderadas",82:"Pancadas violentas",
        95:"Tempestade",96:"Tempestade com granizo",99:"Tempestade severa",
    }
    return tabela.get(wmo, f"Código {wmo}")

def formatar_dados_clima(geo: dict, dados: dict) -> str:
    cur       = dados.get("current", {})
    hora_atual = datetime.now().hour
    hourly    = dados.get("hourly", {})
    prec_prob = hourly.get("precipitation_probability", [])
    temps     = hourly.get("temperature_2m", [])
    wcodes    = hourly.get("weathercode", [])
    proximas  = []
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
    """Geocodifica a cidade, busca o clima e cria a sessão."""
    geo = geocode(req.cidade, req.estado)
    if not geo:
        raise HTTPException(status_code=404, detail="Cidade não encontrada.")

    dados_clima = obter_previsao(geo["lat"], geo["lon"])
    if not dados_clima:
        raise HTTPException(status_code=502, detail="Erro ao obter dados de clima.")

    bloco_clima = formatar_dados_clima(geo, dados_clima)

    # ID de sessão simples baseado em cidade+estado
    session_id = f"{req.cidade.lower()}_{req.estado.lower()}_{datetime.now().timestamp()}"

    # Salva o system prompt junto com a sessão
    sessoes[session_id] = {
        "sistema": f"""Você é um assistente meteorológico inteligente e amigável para {geo['nome']}.
Use os dados abaixo como base para respostas sobre clima:

{bloco_clima}

Regras:
- Responda SEMPRE em português brasileiro.
- Seja conciso e claro; adicione dicas práticas (o que vestir, guarda-chuva, etc.).
- Emita ALERTAS em maiúsculas se houver risco (tempestade, calor extremo, frio intenso).
- Se perguntarem algo fora do clima, responda brevemente e redirecione ao tema.""",
        "historico": [],
        "geo": geo,
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
        "clima":      sessoes[session_id]["clima_atual"],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Recebe mensagem do usuário e retorna resposta do LLM."""
    if req.session_id not in sessoes:
        raise HTTPException(status_code=404, detail="Sessão não encontrada. Reinicie.")

    sessao   = sessoes[req.session_id]
    historico = sessao["historico"]

    historico.append({"role": "user", "content": req.mensagem})

    # Aplica sliding window
    janela = historico[-SLIDING_WINDOW:]

    client = Groq(api_key=GROQ_API_KEY)
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