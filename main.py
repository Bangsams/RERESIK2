import streamlit as st
from PIL import Image
import io
import os
import tempfile
import base64
import requests
from openai import OpenAI

# ===== Tambahan LangSmith =====
from langchain_openai import ChatOpenAI
from langsmith import Client as LangSmithClient
# ==============================

# Ambil secrets dari Streamlit
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PUSHOVER_USER_KEY = st.secrets["PUSHOVER_USER"]
PUSHOVER_APP_TOKEN = st.secrets["PUSHOVER_TOKEN"]
LANGSMITH_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", None)

# Setup environment variables untuk LangChain tracing
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "RERESIK AI"
    ls_client = LangSmithClient(api_key=LANGSMITH_API_KEY)

    # Model LangChain untuk logging
    llm_trace = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title='RERESIK🌴', layout='wide')

# Styling
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #e8f8ef 0%, #f7fff9 100%); color: #003300; }
.title{ font-size:38px; font-weight:700; color:#064e3b; }
.subtitle{ color:#065f46; }
.news-box{ background: white; border-radius:12px; padding:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# Simple page routing
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def go_to_berita():
    if st.session_state.search_input.strip() != '':
        st.session_state.page = 'berita'

        # Logging search ke LangSmith
        if LANGSMITH_API_KEY:
            ls_client.log_event(
                name="search_input",
                description=f"User searched for: {st.session_state.search_input.strip()}",
                metadata={"search_input": st.session_state.search_input.strip()}
            )


# Main page content
if st.session_state.page == 'main':
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="title">RERESIK🌴</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Deteksi sampah & berita tentang kebersihan dan renewable</div>', unsafe_allow_html=True)
    with col2:
        st.text_input(
            'Cari berita atau topik...',
            key='search_input',
            placeholder='Ketik lalu tekan Enter → halaman Berita',
            on_change=go_to_berita
        )

    st.markdown('### Berita Terbaru')
    sample_news = [
        {'title': 'Kegiatan bersih-bersih lingkungan 🧹', 'desc': 'Komunitas lokal mengadakan aksi pemilahan sampah'},
        {'title': 'Tips membuat kompos 🫛', 'desc': 'Langkah mudah membuat kompos dari sampah organik di rumah'},
        {'title': 'Bank Sampah Digital 🖥️', 'desc': 'Inovasi pengelolaan sampah anorganik untuk ekonomi sirkular'}
    ]

    cols = st.columns(3)
    for c, news in zip(cols, sample_news):
        with c:
            st.markdown(
                f'<div class="news-box"><b>{news["title"]}</b><br><small>{news["desc"]}</small></div>',
                unsafe_allow_html=True
            )

    st.markdown('---')
    st.header('Deteksi Sampah - Kamera & Pelaporan')

    img_file = st.camera_input('Arahkan kamera ke tumpukan sampah lalu ambil foto', key='camera_input_unique')

    st.sidebar.header('Kalibrasi Estimasi Berat')
    ref_cm = st.sidebar.number_input('Lebar objek referensi (cm)', min_value=0.0, value=5.0, step=0.1)
    ref_wt = st.sidebar.number_input('Berat objek referensi (gram)', min_value=0.0, value=0.0, step=1.0)
    use_ai = st.sidebar.checkbox('Gunakan OpenAI GPT-4o untuk analisis', value=True)

    def analyze_with_openai(image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        base64_img = base64.b64encode(buf.read()).decode('utf-8')

        # ===== Jika LangSmith aktif, log lewat LangChain =====
        if LANGSMITH_API_KEY:
            messages = [
                ("system", "Anda adalah model vision yang dapat mengklasifikasikan sampah organik atau anorganik dan mengestimasi beratnya."),
                ("human", f"""
Klasifikasikan apakah sampah ini organik atau anorganik. 
Berikan estimasi berat (gram) secara kasar berdasarkan skala visual.
Gambar: data:image/jpeg;base64,{base64_img}
""")
            ]
            result = llm_trace.invoke(messages)
            return result.content
        # ====================================================

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Anda adalah model vision yang dapat mengklasifikasikan sampah organik atau anorganik dan mengestimasi beratnya."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Klasifikasikan apakah sampah ini organik atau anorganik. Berikan estimasi berat (gram) secara kasar berdasarkan skala visual."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
            max_tokens=150
        )
        return resp.choices[0].message.content


    if img_file:
        image = Image.open(img_file)
        st.image(image, caption='Foto diambil', use_column_width=True)

        # Logging saat user mengambil foto
        if LANGSMITH_API_KEY:
            ls_client.log_event(
                name="camera_photo_taken",
                description="User mengambil foto untuk analisis sampah",
                metadata={"use_ai": use_ai}
            )

        if use_ai and OPENAI_API_KEY:
            st.subheader('Hasil Analisis AI')
            try:
                result = analyze_with_openai(image)
                st.write(result)
            except Exception as e:
                st.error(f"Analisis AI gagal: {e}")
        else:
            st.warning("Analisis AI nonaktif atau API key belum diset.")

        if st.button('Download hasil analisis'):
            summary = result if use_ai else "Analisis heuristik."
            b = io.BytesIO(summary.encode())
            st.download_button('Unduh (TXT)', data=b, file_name='analisis.txt')

    st.markdown('---')
    st.header('Laporkan Sampah')

    with st.form('report_form'):
        name = st.text_input('Nama')
        location = st.text_input('Lokasi')
        desc = st.text_area('Deskripsi')
        photo = st.file_uploader('Lampiran (opsional)', type=['jpg', 'png', 'jpeg'])
        submit = st.form_submit_button('Kirim')

        if submit:
            msg = f"Laporan dari {name or 'Anon'}\nLokasi: {location}\nDeskripsi: {desc}"
            attach = None
            if photo:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(photo.getbuffer())
                    attach = tmp.name
            try:
                r = requests.post(
                    'https://api.pushover.net/1/messages.json',
                    data={'token': PUSHOVER_APP_TOKEN, 'user': PUSHOVER_USER_KEY, 'message': msg},
                    files={'attachment': open(attach, 'rb')} if attach else None
                )
                r.raise_for_status()
                st.success('Laporan terkirim.')

                # Logging pengiriman notifikasi ke LangSmith
                if LANGSMITH_API_KEY:
                    ls_client.log_event(
                        name="pushover_notification_sent",
                        description=f"Laporan terkirim: {msg}",
                        metadata={"location": location, "name": name}
                    )

            except Exception as e:
                st.error(f'Gagal mengirim notifikasi: {e}')


if st.session_state.page == 'berita':
    import berita
    berita.show_berita()
