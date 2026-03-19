import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
import markdown
from dateutil.parser import parse

# ==================== CONFIGURACIÓN ====================

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWS_API")
MODEL = "gpt-4o-mini"

MAX_NEWS_DAYS = 30
MAX_ARTICLES_TO_SUMMARIZE = 20
MAX_CHARS_PER_ARTICLE = 6000

RANKING_BLOCK_SIZE = 20
RANKING_SELECT_PER_BLOCK = 5

openai = OpenAI()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (EnterpriseNewsAnalyzer/2.0)"
}

# ==================== PROMPTS ====================

system_prompt_web = """
Eres un analista corporativo senior.

Analiza el contenido de la web de una empresa y genera
un resumen ejecutivo claro, profesional y objetivo.

Estructura:
## Descripción de la empresa
## Propuesta de valor
## Productos / Servicios
## Posicionamiento
## Conclusión ejecutiva
"""

system_prompt_noticias = """
Eres un analista de inteligencia empresarial.

Analiza los siguientes RESÚMENES DE NOTICIAS SELECCIONADAS
y evalúa su impacto estratégico.

NO inventes información.
Marca claramente inferencias.

Estructura:
## Hechos relevantes
## Impacto estratégico
## Señales de futuro
## Conclusión ejecutiva
"""

# ==================== UTILIDADES ====================

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ==================== CLASE WEBSITE ====================

class Website:

    def __init__(self, url_o_busqueda):
        self.url = url_o_busqueda
        self.is_news_search = url_o_busqueda.startswith("NEWS:")
        self.total_news = 0
        self.articles = []
        self.selected_articles = []
        self.summaries = []
        self.text = ""

        if self.is_news_search:
            search_term = url_o_busqueda.replace("NEWS:", "").strip()
            self._load_all_news(search_term)
        else:
            self.text = self._load_web(url_o_busqueda)

    # -------- WEB --------
    def _load_web(self, url):
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "img", "svg", "input"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    # -------- NEWS LOAD --------
    def _load_all_news(self, search_term):
        news = []
        for lang in ["es", "en"]:
            news.extend(self._fetch_newsapi(search_term, lang))
        news.extend(self._fetch_gdelt(search_term))

        unique = {n["url"]: n for n in news if n.get("url")}
        self.articles = sorted(
            unique.values(),
            key=lambda x: parse(x["published"]) if x.get("published") else datetime.min,
            reverse=True
        )

        self.total_news = len(self.articles)

        self.selected_articles = self._ranking_multicall()
        self.summaries = self._summarize_selected()

        self.text = "\n\n".join(f"- {s['summary']}" for s in self.summaries)

    # -------- FETCH NEWSAPI --------
    def _fetch_newsapi(self, search_term, language):
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": search_term,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": 100,
            "from": (datetime.now() - timedelta(days=MAX_NEWS_DAYS)).strftime("%Y-%m-%d"),
            "apiKey": NEWSAPI_KEY
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                return []
            return [{
                "title": a.get("title"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "published": a.get("publishedAt")
            } for a in r.json().get("articles", [])]
        except:
            return []

    # -------- FETCH GDELT --------
    def _fetch_gdelt(self, search_term):
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": search_term,
            "mode": "artlist",
            "timespan": f"{MAX_NEWS_DAYS}d",
            "maxrecords": 200,
            "format": "json"
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                return []
            return [{
                "title": a.get("title"),
                "url": a.get("url"),
                "source": a.get("source"),
                "published": a.get("seendate")
            } for a in r.json().get("articles", [])]
        except:
            return []

    # ==================== RANKING MULTILLAMADA ====================

    def _ranking_multicall(self):
        candidates = self.articles[:]

        while len(candidates) > MAX_ARTICLES_TO_SUMMARIZE:
            winners = []

            for block in chunked(candidates, RANKING_BLOCK_SIZE):
                preview = "\n".join(
                    f"[{i}] {a['title']} | {a['source']} | {a['published']} | {a['url']}"
                    for i, a in enumerate(block)
                )

                prompt = f"""
Eres un analista de inteligencia empresarial.

Selecciona las {RANKING_SELECT_PER_BLOCK} noticias
con MAYOR impacto estratégico.

Devuelve SOLO los índices numéricos.

NOTICIAS:
{preview}
"""

                r = openai.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )

                indices = [
                    int(i.strip())
                    for i in r.choices[0].message.content.replace(",", "\n").split()
                    if i.strip().isdigit()
                ]

                for i in indices[:RANKING_SELECT_PER_BLOCK]:
                    if i < len(block):
                        winners.append(block[i])

            candidates = winners

        return candidates[:MAX_ARTICLES_TO_SUMMARIZE]

    # ==================== RESUMEN ====================

    def _summarize_selected(self):
        summaries = []

        for a in self.selected_articles:
            text = self._fetch_article_text(a["url"])
            if not text:
                continue

            prompt = f"""
Resume esta noticia de forma objetiva.
Incluye solo hechos verificables.

NOTICIA:
{text}
"""

            r = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )

            summaries.append({
                "title": a["title"],
                "url": a["url"],
                "published": a["published"],
                "summary": r.choices[0].message.content
            })

        return summaries

    def _fetch_article_text(self, url):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:MAX_CHARS_PER_ARTICLE]
        except:
            return ""

    def get_contents(self):
        return self.text

# ==================== IA FUNCTIONS ====================

def stream_brochure(company_name, url):
    website = Website(url)
    content = website.get_contents()

    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt_web},
            {"role": "user", "content": f"Empresa: {company_name}\n\n{content}"}
        ],
        stream=True
    )

    return "".join(chunk.choices[0].delta.content or "" for chunk in stream)

def analyze_news(company_name):
    website_news = Website(f"NEWS:{company_name}")
    content = website_news.get_contents()

    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt_noticias},
            {"role": "user", "content": content}
        ],
        stream=True
    )

    analysis = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    return analysis, website_news

# ==================== HTML ====================

def save_report_html(web_md, news_md, website, company_name):
    sources_md = "\n".join(
        f"- **{s['title']}**  \n  🗓️ {s['published']}  \n  🔗 [{s['url']}]({s['url']})"
        for s in website.summaries
    )

    full_markdown = f"""
# Informe de {company_name}

## Análisis corporativo
{web_md}

---

## Noticias analizadas
{sources_md}

---

## Análisis estratégico de noticias
{news_md}
"""

    html_body = markdown.markdown(full_markdown, extensions=["tables", "fenced_code"])

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe {company_name}</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<div class="container">
{html_body}
<div class="timestamp">
Informe generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
</div>
</body>
</html>
"""

    filename = f"{company_name}_informe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    return filename

# ==================== MAIN ====================

def main():
    url = input("🌐 URL empresa: ").strip()
    company = input("🏢 Nombre empresa: ").strip()

    print("⏳ Analizando web...")
    web_report = stream_brochure(company, url)

    print("⏳ Analizando noticias...")
    news_analysis, website_news = analyze_news(company)

    filename = save_report_html(
        web_report,
        news_analysis,
        website_news,
        company
    )

    print("✅ Informe generado:")
    print(os.path.abspath(filename))

if __name__ == "__main__":
    main()
