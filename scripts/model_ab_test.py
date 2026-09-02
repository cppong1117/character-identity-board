#!/usr/bin/env python3
"""
Model A/B Test Framework for Character Identity Board V0.2
Phase 16: Test alternative face recognition models

License Audit (2026-09-02):
- SFace (OpenCV Zoo): Apache 2.0 ✅ Commercial-safe
- ArcFace R50 (InsightFace buffalo_l): Non-commercial research only ❌
- FaceNet (facenet-pytorch): MIT ✅ Commercial-safe
- AdaFace: Need to verify license

Usage:
    python model_ab_test.py --model sface --benchmark benchmark_data.json
    python model_ab_test.py --model facenet --benchmark benchmark_data.json
    python model_ab_test.py --compare --benchmark benchmark_data.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# License Documentation
# ============================================================

MODEL_LICENSES = {
    "sface": {
        "name": "SFace",
        "code_license": "Apache 2.0",
        "weight_license": "Apache 2.0",
        "commercial_use": True,
        "source": "https://github.com/opencv/opencv_zoo",
        "restrictions": "None",
        "embedding_dim": 128,
        "recommended_threshold": 0.50,
    },
    "facenet": {
        "name": "FaceNet (VGGFace2)",
        "code_license": "MIT",
        "weight_license": "MIT (facenet-pytorch)",
        "commercial_use": True,
        "source": "https://github.com/timesler/facenet-pytorch",
        "restrictions": "VGGFace2 training data license may apply",
        "embedding_dim": 512,
        "recommended_threshold": 0.80,
    },
    "arcface": {
        "name": "ArcFace R50 (InsightFace buffalo_l)",
        "code_license": "MIT",
        "weight_license": "Non-commercial research only",
        "commercial_use": False,  # Need commercial license
        "source": "https://github.com/deepinsight/insightface",
        "restrictions": "Pretrained weights require commercial license from insightface.ai",
        "embedding_dim": 512,
        "recommended_threshold": 0.35,
    },
}


# ============================================================
# Base Face Recognizer
# ============================================================

class FaceRecognizer:
    """Base class for face recognizers."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_embedding(self, img_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """
        Extract face embedding from image and detection.
        
        Args:
            img_bgr: BGR image
            detection: Dict with 'bbox', 'landmarks', 'score'
        
        Returns:
            Normalized embedding vector or None if failed
        """
        raise NotImplementedError


class SFaceRecognizer(FaceRecognizer):
    """SFace recognizer using OpenCV."""
    
    def __init__(self):
        super().__init__("sface")
        model_path = str(Path.home() / "character-identity-board-data/cache/models/face_recognition_sface.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SFace model not found: {model_path}")
        self.sface = cv2.FaceRecognizerSF.create(model_path, "")
    
    def get_embedding(self, img_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        bbox = detection['bbox']
        lm = detection.get('landmarks')
        score = detection.get('score', 0.8)
        
        # If landmarks available, use proper alignment
        if lm and len(lm) == 5:
            face_arr = np.asarray(
                bbox + [coord for pt in lm for coord in pt] + [score],
                dtype=np.float32,
            )
            aligned = self.sface.alignCrop(img_bgr, face_arr)
        else:
            # Fallback: crop face with margin
            x, y, w, h = bbox
            margin = 0.3
            x0 = max(0, int(x - w * margin))
            y0 = max(0, int(y - h * margin))
            x1 = min(img_bgr.shape[1], int(x + w * (1 + margin)))
            y1 = min(img_bgr.shape[0], int(y + h * (1 + margin)))
            face_crop = img_bgr[y0:y1, x0:x1]
            if face_crop.size == 0:
                return None
            aligned = cv2.resize(face_crop, (112, 112))
        
        feat = self.sface.feature(aligned)
        
        # Flatten to 1D
        feat = feat.flatten()
        
        # Normalize
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        
        return feat.astype(np.float32)


class FaceNetRecognizer(FaceRecognizer):
    """FaceNet recognizer using facenet-pytorch."""
    
    def __init__(self):
        super().__init__("facenet")
        try:
            from facenet_pytorch import InceptionResnetV1
            import torch
            
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except ImportError:
            raise ImportError("facenet-pytorch not installed. Run: pip install facenet-pytorch")
    
    def get_embedding(self, img_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        import torch
        from torchvision import transforms
        
        bbox = detection['bbox']
        lm = detection['landmarks']
        score = detection['score']
        
        # Extract face crop with margin
        x, y, w, h = bbox
        margin = 0.2
        x0 = max(0, int(x - w * margin))
        y0 = max(0, int(y - h * margin))
        x1 = min(img_bgr.shape[1], int(x + w * (1 + margin)))
        y1 = min(img_bgr.shape[0], int(y + h * (1 + margin)))
        
        face_crop = img_bgr[y0:y1, x0:x1]
        if face_crop.size == 0:
            return None
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to 160x160 (FaceNet input)
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face_resized).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        # Normalize
        embedding = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)


# ============================================================
# Model Factory
# ============================================================

def create_recognizer(model_name: str) -> FaceRecognizer:
    """Create face recognizer by name."""
    if model_name == "sface":
        return SFaceRecognizer()
    elif model_name == "facenet":
        return FaceNetRecognizer()
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Benchmark Runner
# ============================================================

class BenchmarkRunner:
    """Run face recognition benchmark."""
    
    def __init__(self, recognizer: FaceRecognizer, benchmark_data: dict):
        self.recognizer = recognizer
        self.benchmark_data = benchmark_data
        self.results = []
    
    def run(self) -> dict:
        """Run benchmark and return metrics."""
        print(f"Running benchmark for {self.recognizer.name}...")
        start_time = time.time()
        
        same_person_scores = []
        different_person_scores = []
        
        for pair in self.benchmark_data.get("pairs", []):
            # Load images
            img1 = cv2.imread(pair["image1"])
            img2 = cv2.imread(pair["image2"])
            
            if img1 is None or img2 is None:
                continue
            
            # Get embeddings
            emb1 = self.recognizer.get_embedding(img1, pair["detection1"])
            emb2 = self.recognizer.get_embedding(img2, pair["detection2"])
            
            if emb1 is None or emb2 is None:
                continue
            
            # Compute cosine similarity
            similarity = float(np.dot(emb1, emb2))
            
            # Record result
            result = {
                "pair_id": pair["pair_id"],
                "same_person": pair["same_person"],
                "similarity": similarity,
            }
            self.results.append(result)
            
            if pair["same_person"]:
                same_person_scores.append(similarity)
            else:
                different_person_scores.append(similarity)
        
        elapsed = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_metrics(same_person_scores, different_person_scores)
        metrics["elapsed_seconds"] = elapsed
        metrics["model"] = self.recognizer.name
        metrics["total_pairs"] = len(self.results)
        metrics["same_person_pairs"] = len(same_person_scores)
        metrics["different_person_pairs"] = len(different_person_scores)
        
        return metrics
    
    def _compute_metrics(
        self,
        same_person_scores: List[float],
        different_person_scores: List[float],
    ) -> dict:
        """Compute precision, recall, FAR, FRR at various thresholds."""
        if not same_person_scores or not different_person_scores:
            return {"error": "No valid pairs"}
        
        thresholds = np.arange(0.10, 0.95, 0.05)
        metrics = {"thresholds": []}
        
        for threshold in thresholds:
            tp = sum(1 for s in same_person_scores if s >= threshold)
            fn = sum(1 for s in same_person_scores if s < threshold)
            fp = sum(1 for s in different_person_scores if s >= threshold)
            tn = sum(1 for s in different_person_scores if s < threshold)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics["thresholds"].append({
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "far": float(far),
                "frr": float(frr),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            })
        
        # Add distribution stats
        metrics["same_person"] = {
            "mean": float(np.mean(same_person_scores)),
            "median": float(np.median(same_person_scores)),
            "std": float(np.std(same_person_scores)),
            "min": float(np.min(same_person_scores)),
            "max": float(np.max(same_person_scores)),
        }
        metrics["different_person"] = {
            "mean": float(np.mean(different_person_scores)),
            "median": float(np.median(different_person_scores)),
            "std": float(np.std(different_person_scores)),
            "min": float(np.min(different_person_scores)),
            "max": float(np.max(different_person_scores)),
        }
        
        return metrics


# ============================================================
# Report Generator
# ============================================================

def generate_report(
    model_results: Dict[str, dict],
    output_dir: str,
) -> str:
    """Generate A/B test report."""
    report_path = os.path.join(output_dir, "model_ab_test_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model A/B Test Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # License summary
        f.write("## License Audit\n\n")
        f.write("| Model | Code License | Weight License | Commercial Use |\n")
        f.write("|-------|--------------|----------------|----------------|\n")
        for name, info in MODEL_LICENSES.items():
            commercial = "✅ Yes" if info["commercial_use"] else "❌ No"
            f.write(f"| {info['name']} | {info['code_license']} | {info['weight_license']} | {commercial} |\n")
        
        # Model results
        f.write("\n## Benchmark Results\n\n")
        
        for model_name, metrics in model_results.items():
            f.write(f"### {MODEL_LICENSES.get(model_name, {}).get('name', model_name)}\n\n")
            
            if "error" in metrics:
                f.write(f"Error: {metrics['error']}\n\n")
                continue
            
            # Distribution
            f.write("#### Score Distribution\n\n")
            f.write("| Metric | Same Person | Different Person |\n")
            f.write("|--------|-------------|------------------|\n")
            sp = metrics["same_person"]
            dp = metrics["different_person"]
            f.write(f"| Mean | {sp['mean']:.4f} | {dp['mean']:.4f} |\n")
            f.write(f"| Median | {sp['median']:.4f} | {dp['median']:.4f} |\n")
            f.write(f"| Std | {sp['std']:.4f} | {dp['std']:.4f} |\n")
            f.write(f"| Min | {sp['min']:.4f} | {dp['min']:.4f} |\n")
            f.write(f"| Max | {sp['max']:.4f} | {dp['max']:.4f} |\n")
            
            # Threshold table
            f.write("\n#### Threshold Performance\n\n")
            f.write("| Threshold | Precision | Recall | FAR | FRR | TP | FN | FP | TN |\n")
            f.write("|-----------|-----------|--------|-----|-----|----|----|----|----|\n")
            for t in metrics["thresholds"]:
                f.write(
                    f"| {t['threshold']:.2f} | {t['precision']:.4f} | {t['recall']:.4f} | "
                    f"{t['far']:.4f} | {t['frr']:.4f} | {t['tp']} | {t['fn']} | {t['fp']} | {t['tn']} |\n"
                )
            
            f.write(f"\n**Total pairs**: {metrics['total_pairs']}\n")
            f.write(f"**Elapsed**: {metrics['elapsed_seconds']:.1f}s\n\n")
        
        # Comparison
        if len(model_results) > 1:
            f.write("## Model Comparison\n\n")
            f.write("| Model | Same-Person Mean | Different-Person Mean | Separation | Best Precision@99% Recall |\n")
            f.write("|-------|------------------|----------------------|------------|--------------------------|\n")
            
            for model_name, metrics in model_results.items():
                if "error" in metrics:
                    continue
                sp_mean = metrics["same_person"]["mean"]
                dp_mean = metrics["different_person"]["mean"]
                separation = sp_mean - dp_mean
                
                # Find best precision at >=99% recall
                best_prec = 0
                for t in metrics["thresholds"]:
                    if t["recall"] >= 0.99 and t["precision"] > best_prec:
                        best_prec = t["precision"]
                
                f.write(f"| {model_name} | {sp_mean:.4f} | {dp_mean:.4f} | {separation:.4f} | {best_prec:.4f} |\n")
        
        # Recommendation
        f.write("\n## Recommendation\n\n")
        f.write("Based on license audit and benchmark results:\n\n")
        
        # Find best commercial-safe model
        best_commercial = None
        best_separation = -1
        for model_name, metrics in model_results.items():
            if MODEL_LICENSES.get(model_name, {}).get("commercial_use", False):
                if "error" not in metrics:
                    separation = metrics["same_person"]["mean"] - metrics["different_person"]["mean"]
                    if separation > best_separation:
                        best_separation = separation
                        best_commercial = model_name
        
        if best_commercial:
            f.write(f"**Recommended**: {MODEL_LICENSES[best_commercial]['name']}\n")
            f.write(f"- Separation: {best_separation:.4f}\n")
            f.write(f"- License: Commercial-safe\n")
        else:
            f.write("**No commercial-safe model found.**\n")
    
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Model A/B Test for CIB")
    parser.add_argument("--model", choices=["sface", "facenet"], help="Single model to test")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON file")
    parser.add_argument("--output", default="reports/v02_accuracy_recovery", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Compare all commercial-safe models")
    args = parser.parse_args()
    
    # Load benchmark data
    with open(args.benchmark, "r") as f:
        benchmark_data = json.load(f)
    
    os.makedirs(args.output, exist_ok=True)
    
    model_results = {}
    
    if args.compare:
        # Test all commercial-safe models
        for model_name in ["sface", "facenet"]:
            try:
                recognizer = create_recognizer(model_name)
                runner = BenchmarkRunner(recognizer, benchmark_data)
                metrics = runner.run()
                model_results[model_name] = metrics
                print(f"✅ {model_name}: separation={metrics.get('same_person', {}).get('mean', 0) - metrics.get('different_person', {}).get('mean', 0):.4f}")
            except Exception as e:
                print(f"❌ {model_name}: {e}")
                model_results[model_name] = {"error": str(e)}
    elif args.model:
        # Test single model
        recognizer = create_recognizer(args.model)
        runner = BenchmarkRunner(recognizer, benchmark_data)
        metrics = runner.run()
        model_results[args.model] = metrics
    else:
        print("Error: specify --model or --compare")
        sys.exit(1)
    
    # Generate report
    report_path = generate_report(model_results, args.output)
    print(f"\nReport saved to: {report_path}")
    
    # Save raw metrics
    metrics_path = os.path.join(args.output, "model_ab_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
