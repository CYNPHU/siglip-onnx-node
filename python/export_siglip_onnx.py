"""
One-time export: siglip-base image encoder -> ONNX so it can run in Node via
onnxruntime-node. Also prints the exact preprocessing config (size, mean, std,
resample) needed to replicate the processor in JS, and verifies the ONNX output
matches the PyTorch model (cosine ~1.0). Run once:
  python python/export_siglip_onnx.py
"""
import json, os, numpy as np, torch
from transformers import AutoProcessor, AutoModel

MODEL = "google/siglip-base-patch16-224"
OUT = os.path.join("models", "siglip_vision.onnx")

class ImgEncoder(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values):
        f = self.m.get_image_features(pixel_values=pixel_values)
        if hasattr(f, "pooler_output"): f = f.pooler_output  # match precompute
        return f

def main():
    os.makedirs("models", exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    ip = proc.image_processor
    cfg = {"size": ip.size, "image_mean": ip.image_mean, "image_std": ip.image_std,
           "resample": ip.resample, "do_rescale": ip.do_rescale,
           "rescale_factor": getattr(ip, "rescale_factor", 1/255), "do_normalize": ip.do_normalize}
    print("PREPROCESS CONFIG (replicate in JS):"); print(json.dumps(cfg, default=str, indent=2))

    wrap = ImgEncoder(model).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(wrap, (dummy,), OUT, input_names=["pixel_values"], output_names=["image_features"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
                      opset_version=18, do_constant_folding=True, dynamo=False)
    print(f"\nexported -> {OUT}")

    # parity: torch vs onnxruntime on the same random input
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        t = wrap(x).numpy()
    o = sess.run(["image_features"], {"pixel_values": x.numpy()})[0]
    tn = t / np.linalg.norm(t, axis=1, keepdims=True)
    on = o / np.linalg.norm(o, axis=1, keepdims=True)
    cos = float((tn * on).sum(1).mean())
    maxabs = float(np.abs(t - o).max())
    print(f"\nPARITY torch vs onnx: cosine={cos:.6f}  max|diff|={maxabs:.6f}  (want cos~1.0)")
    print("OK" if cos > 0.9999 else "MISMATCH — investigate")

if __name__ == "__main__":
    main()
