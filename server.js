/**
 * server.js  –  Express server (Docker-aware version)
 *
 * Instead of exec("python run_yolo.py …"), it forwards the uploaded
 * file to the Python Flask container via HTTP and passes the JSON
 * result straight to the browser.
 */

const express  = require('express');
const multer   = require('multer');
const cors     = require('cors');
const path     = require('path');
const fs       = require('fs');
const FormData = require('form-data');
const fetch    = (...args) => import('node-fetch').then(({ default: f }) => f(...args));

const app  = express();
const PORT = 3000;

// Python API endpoint – use env var so it works both locally and in Docker
const PYTHON_API = process.env.PYTHON_API_URL || 'http://localhost:5000';

// ── Middleware ────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ── Upload dir ───────────────────────────────────────────────────
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, 'uploads/'),
    filename:    (_req,  file, cb) => {
        const safe = file.originalname.replace(/\s+/g, '_');
        cb(null, Date.now() + '_' + safe);
    }
});
const upload = multer({ storage });

// ── Detection endpoint ───────────────────────────────────────────
app.post('/api/detect', upload.single('mediaUpload'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'Tidak ada file yang diunggah!' });
    }

    const filePath = req.file.path;
    const fileName = req.file.filename;
    console.log(`[Node.js] File diterima: ${filePath}`);

    try {
        // Forward file to Python AI service
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath), req.file.originalname);

        const pyRes  = await fetch(`${PYTHON_API}/predict`, { method: 'POST', body: form });
        const result = await pyRes.json();

        if (!pyRes.ok || result.error) {
            console.error('[Python API error]', result);
            return res.status(500).json({ error: result.error || 'Gagal memproses dengan AI.' });
        }

        console.log('[Python result]', result);

        res.json({
            status:     'success',
            isFake:     result.prediction === 'FAKE',
            score:      Math.round(result.isFake ? result.fake_ratio : result.real_ratio),
            fake_ratio: result.fake_ratio,
            real_ratio: result.real_ratio,
            fileName,
        });

    } catch (err) {
        console.error('[Node.js] Fetch error:', err.message);
        res.status(500).json({ error: 'Tidak dapat menghubungi layanan AI.' });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Server berjalan di http://localhost:${PORT}`);
    console.log(`🐍 Python API target: ${PYTHON_API}`);
});
