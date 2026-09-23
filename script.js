document.addEventListener("DOMContentLoaded", () => {
    // 1. File Upload Interaction
    const fileInput = document.getElementById("mediaUpload");
    const uploadArea = document.getElementById("uploadArea");
    const filePreview = document.getElementById("filePreview");
    const fileNameSpan = document.getElementById("fileName");

    if (fileInput && uploadArea) {
        fileInput.addEventListener("change", function() {
            if (this.files && this.files.length > 0) {
                const fileName = this.files[0].name;
                fileNameSpan.textContent = fileName;
                uploadArea.classList.add("hidden");
                filePreview.classList.remove("hidden");
            }
        });
    }

    // 2. Scan Backend Integration
    const scanForm = document.getElementById("scanForm");
    const loader = document.getElementById("loader");
    const scanBtn = document.getElementById("scanBtn");

    if (scanForm) {
        scanForm.addEventListener("submit", async (e) => {
            e.preventDefault(); 
            
            if(fileInput && fileInput.files.length === 0) return;

            // Sembunyikan tombol, tampilkan loader
            if(filePreview) filePreview.classList.add("hidden");
            scanBtn.classList.add("hidden");
            loader.classList.remove("hidden");

            // Siapkan file untuk dikirim ke Backend
            const formData = new FormData();
            formData.append("mediaUpload", fileInput.files[0]);

            try {
                // Tembak API Node.js (pastikan server.js sedang jalan)
                const response = await fetch("http://localhost:3000/api/detect", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    // Berhasil! Lempar data AI ke halaman result.html
                    window.location.href = `result.html?score=${data.score}&isfake=${data.isFake}&file=${data.fileName}`;
                } else {
                    alert("Terjadi kesalahan di server: " + data.error);
                    // Kembalikan UI seperti semula
                    loader.classList.add("hidden");
                    scanBtn.classList.remove("hidden");
                }
            } catch (error) {
                console.error("Gagal menghubungi Backend:", error);
                alert("Gagal menghubungi server Backend. Pastikan Node.js menyala.");
                loader.classList.add("hidden");
                scanBtn.classList.remove("hidden");
            }
        });
    }

    // 3. Logic for Results page rendering
    const resultCard = document.getElementById("resultCard");
    if (resultCard) {
        const urlParams = new URLSearchParams(window.location.search);
        const score = parseInt(urlParams.get('score'));
        const isFake = urlParams.get('isfake') === 'true';
        const fileName = urlParams.get('file'); // Tangkap nama file

        // Logika untuk menampilkan Media (Foto/Video)
        if (fileName) {
            const mediaContainer = document.querySelector(".result-image");
            const isVideo = fileName.match(/\.(mp4|avi|mov|webm)$/i);
            
            if (isVideo) {
                // Jika video, ganti tag <img> menjadi <video>
                if (mediaContainer) {
                    mediaContainer.innerHTML = `<video autoplay loop muted controls style="width: 100%; border-radius: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.5);"><source src="uploads/${fileName}" type="video/mp4">Browser Anda tidak mendukung video.</video>`;
                }
            } else {
                // Jika gambar, masukkan ke tag <img>
                const imgEl = document.getElementById("uploadedImage");
                if (imgEl) {
                    imgEl.src = "uploads/" + fileName;
                }
            }
        }

        if (!isNaN(score)) {
            let realPct = isFake ? (100 - score) : score;
            let fakePct = isFake ? score : (100 - score);

            const realEl = document.getElementById("realPercentage");
            const fakeEl = document.getElementById("fakePercentage");
            if(realEl) realEl.innerText = realPct + "%";
            if(fakeEl) fakeEl.innerText = fakePct + "%";

            const verdictBadge = document.getElementById("verdictBadge");
            const verdictText = document.getElementById("verdictText");

            if (verdictBadge && verdictText) {
                verdictBadge.className = "verdict-badge";
                
                if (isFake) {
                    if (fakePct >= 80) {
                        verdictText.innerText = "Definitely Fake";
                        verdictBadge.classList.add("definitely-fake");
                    } else {
                        verdictText.innerText = "Likely Fake";
                        verdictBadge.classList.add("likely-fake");
                    }
                } else {
                    if (realPct >= 80) {
                        verdictText.innerText = "Definitely Real";
                        verdictBadge.classList.add("definitely-real");
                    } else {
                        verdictText.innerText = "Likely Real";
                        verdictBadge.classList.add("likely-real");
                    }
                }
            }
        }
    }
});