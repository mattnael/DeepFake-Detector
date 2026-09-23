"""
Deepfake Detection using YOLO - Inference Script

This script runs inference on videos using a trained YOLO model.

Output: results/yolo/
"""

import os
import sys
import torch
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_loader import VideoFrameExtractor, load_testing_videos


class YOLOInference:
    """Inference class for YOLO-based deepfake detection."""
    
    def __init__(self, model_path: str = "output/yolo/best_yolo.pt"):
        """
        Initialize YOLO inference.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path("results/yolo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """Load the trained YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            print(f"Model loaded from: {self.model_path}")
        except ImportError:
            print("Ultralytics not installed. Run: pip install ultralytics")
            raise
    
    def predict_video(self, video_path: str, frames_to_process: int = 10) -> dict:
        """
        Predict if a media (video/image) is deepfake or real.
        
        Args:
            video_path: Path to media file
            frames_to_process: Number of frames to process (if video)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            self.load_model()
        
        # ====================================================
        # LOGIKA BARU: Deteksi otomatis Foto atau Video
        # ====================================================
        file_ext = Path(video_path).suffix.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        if file_ext in image_extensions:
            # JIKA FOTO: Baca foto sebagai 1 frame
            frame = cv2.imread(video_path)
            frames = [frame] if frame is not None else []
        else:
            # JIKA VIDEO: Gunakan mesin ekstraktor video
            frame_extractor = VideoFrameExtractor(frames_to_process)
            frames = frame_extractor.extract_from_video(video_path)
        # ====================================================
        
        if len(frames) == 0:
            return {
                "video_path": video_path,
                "error": "Could not extract frames from media",
                "prediction": "unknown",
                "confidence": 0.0
            }
        
        # Predict on each frame
        predictions = []
        for frame in frames:
            results = self.model(frame, verbose=False)
            
            # Get classification result
            probs = results[0].probs
            pred_class = probs.top1
            pred_conf = probs.top1conf.item()
            
            predictions.append({
                'class': 'real' if pred_class == 1 else 'fake',
                'confidence': pred_conf
            })
        
        # Aggregate predictions
        fake_count = sum(1 for p in predictions if p['class'] == 'fake')
        real_count = sum(1 for p in predictions if p['class'] == 'real')
        total_frames = len(predictions)
        
        fake_ratio = (fake_count / total_frames) * 100
        real_ratio = (real_count / total_frames) * 100
        
        if fake_ratio >= 80:
            status = "Dipastikan Fake"
            final_prediction = 'fake'
        elif fake_ratio >= 60:
            status = "Likely Fake"
            final_prediction = 'fake'
        elif fake_ratio > 40:
            status = "Ambigu/Ragu"
            final_prediction = 'unknown'
        elif fake_ratio >= 20:
            status = "Likely Real"
            final_prediction = 'real'
        else:
            status = "Dipastikan Real"
            final_prediction = 'real'
            
        # Tie-breaker if Ambigu
        if status == "Ambigu/Ragu":
            fake_conf = np.mean([p['confidence'] for p in predictions if p['class'] == 'fake']) if fake_count > 0 else 0
            real_conf = np.mean([p['confidence'] for p in predictions if p['class'] == 'real']) if real_count > 0 else 0
            final_prediction = 'fake' if fake_conf > real_conf else 'real'
        
        avg_confidence = float(np.mean([p['confidence'] for p in predictions]))
        
        return {
            "video_path": video_path,
            "video_name": Path(video_path).name,
            "prediction": final_prediction,
            "status": status,
            "fake_ratio": fake_ratio,
            "real_ratio": real_ratio,
            "confidence": avg_confidence,
            "fake_frames": fake_count,
            "real_frames": real_count,
            "frame_predictions": predictions
        }
    
    def predict_batch(self, video_paths: list) -> list:
        """
        Predict on a batch of videos.
        
        Args:
            video_paths: List of video paths
            
        Returns:
            List of prediction results
        """
        results = []
        for video_path in video_paths:
            print(f"Processing: {video_path}")
            result = self.predict_video(video_path)
            results.append(result)
            
            # Print result
            if "error" not in result:
                print(f"  Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
            else:
                print(f"  Error: {result['error']}")
        
        return results
    
    def save_results(self, results: list, output_file: str = None):
        """
        Save results to file.
        
        Args:
            results: List of prediction results
            output_file: Output file path
        """
        if output_file is None:
            output_file = self.output_dir / "predictions.txt"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Deepfake Detection - YOLO Results\n")
            f.write("=" * 60 + "\n\n")
            
            for result in results:
                f.write(f"Video: {result.get('video_name', 'unknown')}\n")
                if "error" in result:
                    f.write(f"  Error: {result['error']}\n")
                else:
                    f.write(f"  Prediction: {result['prediction']}\n")
                    f.write(f"  Confidence: {result['confidence']:.4f}\n")
                    f.write(f"  Fake Frames: {result['fake_frames']}\n")
                    f.write(f"  Real Frames: {result['real_frames']}\n")
                f.write("-" * 40 + "\n")
            
            # Summary
            total = len([r for r in results if 'error' not in r])
            fake_count = sum(1 for r in results if r.get('prediction') == 'fake')
            real_count = sum(1 for r in results if r.get('prediction') == 'real')
            
            f.write("\nSummary:\n")
            f.write(f"  Total videos processed: {total}\n")
            f.write(f"  Predicted as Fake: {fake_count}\n")
            f.write(f"  Predicted as Real: {real_count}\n")
        
        print(f"Results saved to: {output_file}")
        return output_file


def main():
    """Main inference function."""
    import argparse
    parser = argparse.ArgumentParser(description="Deepfake Detection - YOLO Inference")
    parser.add_argument("video_path", type=str, nargs='?', help="Path spesifik ke video yang ingin dites")
    args = parser.parse_args()

    # Configuration
    MODEL_PATH = "model/best.pt"
    FRAMES_PER_VIDEO = 10
    
    print("=" * 60)
    print("Deepfake Detection - YOLO Inference")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    
    # Initialize inference
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run training first: python train/train_yolo.py")
        return
        
    inference = YOLOInference(model_path=MODEL_PATH)
    
    if args.video_path:
        video_target = args.video_path
        if not Path(video_target).exists():
            print(f"Video {video_target} tidak ditemukan!")
            return
            
        print(f"Menguji Video: {video_target}\n")
        hasil = inference.predict_video(video_target, frames_to_process=FRAMES_PER_VIDEO)
        
        if "error" in hasil:
            print(f"Error: {hasil['error']}")
            return
            
        print("-" * 45)
        print("KESIMPULAN AKHIR:")
        print(f"Status Kategori   : {hasil.get('status', 'Unknown')}")
        print(f"Prediksi Dominan  : {hasil['prediction'].upper()}")
        print(f"Rata-rata Conf    : {hasil['confidence'] * 100:.2f}%")
        print("\nDetail Frame yang Diekstrak:")
        print(f"> Frame Real: {hasil['real_frames']} ({hasil.get('real_ratio', 0):.1f}%)")
        print(f"> Frame Fake: {hasil['fake_frames']} ({hasil.get('fake_ratio', 0):.1f}%)\n")
        
        for i, frame in enumerate(hasil['frame_predictions']):
            print(f"Frame {i+1:02d}: {frame['class'].upper()} (Conf: {frame['confidence'] * 100:.2f}%)")
        print("-" * 45)
        
    else:
        print("\nCARA PENGGUNAAN BARU:")
        print("Ketik di terminal dengan target path video Anda di akhir:")
        print(r"python run/run_yolo.py dataset\Celeb-real\id0_0000.mp4")
        print("=" * 60)


if __name__ == "__main__":
    main()
