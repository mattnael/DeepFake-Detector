# 🚀 DeepFake Detector – Docker Deployment Guide

Jalankan seluruh aplikasi dengan **satu perintah** — tidak perlu hosting, tidak perlu domain, tidak perlu biaya.

---

## Prasyarat (hanya sekali)

Install **Docker Desktop** sesuai OS:

| OS | Link |
|----|------|
| Windows | https://docs.docker.com/desktop/install/windows-install/ |
| macOS   | https://docs.docker.com/desktop/install/mac-install/ |
| Linux   | https://docs.docker.com/desktop/install/linux-install/ |

Setelah install, pastikan Docker sudah berjalan (ada ikon paus di taskbar/menubar).

---

## Struktur File yang Diperlukan

Pastikan folder proyek Anda terlihat seperti ini sebelum build:

```
DeepfakeDetectorWeb/
├── Dockerfile.node          ← (file baru)
├── Dockerfile.python        ← (file baru)
├── docker-compose.yml       ← (file baru)
├── .dockerignore            ← (file baru)
├── python_api.py            ← (file baru)
├── server.js                ← (ganti dengan versi baru)
│
├── index.html
├── detector.html
├── about.html
├── result.html
├── style.css
├── script.js
├── run_yolo.py
├── package.json
├── requirements.txt
├── model/
│   └── best.pt
└── utils/
    └── data_loader.py
```

---

## Langkah Deploy (Step by Step)

### Langkah 1 — Salin file Docker ke folder proyek

Salin semua file berikut ke dalam folder root proyek (`DeepfakeDetectorWeb/`):
- `Dockerfile.node`
- `Dockerfile.python`
- `docker-compose.yml`
- `.dockerignore`
- `python_api.py`
- `server.js` (gantikan yang lama)

### Langkah 2 — Tambah dependensi Node (node-fetch & form-data)

Buka terminal di folder proyek, jalankan:

```bash
npm install node-fetch form-data
```

### Langkah 3 — Build Docker image

```bash
docker compose build
```

> ⏳ Proses ini memakan waktu 5–15 menit pertama kali (mengunduh Python, Node, dan semua library AI).
> Build berikutnya jauh lebih cepat karena cache.

### Langkah 4 — Jalankan aplikasi

```bash
docker compose up
```

Tunggu hingga muncul pesan:
```
deepfake-python  | [Python API] Model ready. Listening on :5000
deepfake-node    | 🚀 Server berjalan di http://localhost:3000
```

### Langkah 5 — Buka di browser

Buka: **http://localhost:3000**

Selesai! 🎉

---

## Cara Menghentikan Aplikasi

```bash
# Di terminal yang sama, tekan:
Ctrl + C

# Atau dari terminal lain:
docker compose down
```

---

## Menjalankan di Background (tanpa terminal terbuka)

```bash
docker compose up -d        # start di background
docker compose down         # stop
docker compose logs -f      # lihat log kapan saja
```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Port 3000 sudah dipakai | Ganti `"3000:3000"` menjadi `"3001:3000"` di `docker-compose.yml` |
| Docker tidak bisa jalan | Pastikan Docker Desktop sudah dibuka (ada ikon paus) |
| Build gagal saat download | Coba lagi — biasanya masalah koneksi sementara |
| Python API belum siap | Tunggu beberapa detik, model AI perlu waktu load saat pertama kali |
| `node-fetch` error | Jalankan `npm install node-fetch form-data` lalu build ulang |

---

## Catatan Teknis

- **Tidak ada GPU diperlukan** — Docker menggunakan CPU untuk inferensi (cukup untuk demo/testing).
- Uploaded files disimpan di Docker volume `uploads_data` (tidak hilang saat restart).
- Python container tidak bisa diakses langsung dari browser — hanya Node.js yang berkomunikasi dengannya melalui jaringan internal Docker.
