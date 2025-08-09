import streamlit as st
import requests
from openai import OpenAI

# Ambil API key dari streamlit secrets
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def cari_berita(query):
    if not SERPER_API_KEY:
        st.error("API Key SERPER_API_KEY tidak ditemukan di Streamlit secrets")
        return []

    url = "https://google.serper.dev/news"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        berita_list = []
        for item in data.get("news", []):
            berita_list.append({
                "judul": item.get("title", ""),
                "link": item.get("link", ""),
                "sumber": item.get("source", ""),
                "thumbnail": item.get("thumbnail", None)
            })

        return berita_list

    except requests.exceptions.RequestException as e:
        st.error(f"Error saat request API: {e}")
        return []
    except ValueError:
        st.error("Gagal parsing response JSON.")
        return []

def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Gagal membuat ringkasan: {e}")
        return text

def show_berita():
    st.title("Halaman Berita")

    keyword = st.session_state.get('search_input', '').strip()
    if not keyword:
        st.info("Masukkan kata kunci pencarian di halaman utama untuk mencari berita.")
        return

    st.write(f"Hasil pencarian untuk: **{keyword}**")

    berita_list = cari_berita(keyword)
    if not berita_list:
        st.warning("Berita tidak ditemukan atau terjadi kesalahan.")
        return

    # Tampilkan thumbnail pertama yang ada
    first_thumbnail_shown = False
    for b in berita_list:
        if b['thumbnail'] and not first_thumbnail_shown:
            st.image(b['thumbnail'], use_column_width=True)
            first_thumbnail_shown = True
            break

    # Batasi maksimal 5 berita
    for b in berita_list[:5]:
        st.markdown(f"### [{b['judul']}]({b['link']})")
        st.markdown(f"**Sumber:** {b['sumber']}")

        prompt = (
            f"Buat ringkasan singkat 1 paragraf dari judul berita berikut ini:\n\n"
            f"Judul: {b['judul']}\n"
            f"Sumber: {b['sumber']}\n\n"
            "Ringkasan:"
        )
        teks_ringkas = summarize_text(prompt)
        st.write(teks_ringkas)

        st.markdown("---")
