
import os, glob, argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# PARAMETERS
# =========================================================
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.40
MIN_ASPECT = 1.5
MAX_ASPECT = 6.5
TOP_K_CANDIDATES = 6

COVERAGE_THRESHOLD = 0.18
EDGE_DENSITY_EARLY_TH = 0.045

LABEL_IF_NO_PLATE_FOUND = 0

# =========================================================
# UTILITIES
# =========================================================
def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")
    imgs = []
    for e in exts:
        imgs += glob.glob(os.path.join(folder, e))
    return sorted(imgs)

# =========================================================
# PLATE DETECTION
# =========================================================
def find_plate_candidates(img):
    H, W = img.shape[:2]
    A = H * W

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 9, 75, 75)

    e = cv2.Canny(g, 60, 180)
    e = cv2.dilate(e, np.ones((3,3), np.uint8))

    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / (h + 1e-6)

        if area < MIN_AREA_RATIO*A or area > MAX_AREA_RATIO*A:
            continue
        if MIN_ASPECT <= ar <= MAX_ASPECT:
            cand.append((x,y,w,h))

    cand.sort(key=lambda r: r[2]*r[3], reverse=True)
    return cand[:TOP_K_CANDIDATES]

# =========================================================
# COVERAGE SCORING
# =========================================================
def coverage_score_and_edges(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = g.shape[:2]
    s = 320 / max(w, 1)
    g = cv2.resize(g, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(g, 50, 150)
    edge_density = edges.mean() / 255.0

    blur = cv2.GaussianBlur(g, (5,5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    blob_ratio = th.mean() / 255.0

    contrast = g.std() / 255.0

    score = 0.0
    score += max(0.0, 0.07 - edge_density) * 4.0
    score += max(0.0, blob_ratio - 0.18) * 1.5
    score += max(0.0, 0.16 - contrast) * 2.0

    return score, edge_density

# =========================================================
# CLASSIFIER
# =========================================================
def is_covered(img, score_list):
    cands = find_plate_candidates(img)
    if not cands:
        score_list.append(0.0)
        return LABEL_IF_NO_PLATE_FOUND

    for x,y,w,h in cands:
        roi = img[y:y+h, x:x+w]
        score, edge_density = coverage_score_and_edges(roi)
        score_list.append(score)

        if edge_density < EDGE_DENSITY_EARLY_TH:
            return 1
        if score >= COVERAGE_THRESHOLD:
            return 1

    return 0

# =========================================================
# MAIN
# =========================================================
def main(inp, out):
    os.makedirs("results", exist_ok=True)

    y_true, y_pred, scores = [], [], []

    covered_files = set()
    if os.path.exists("dataset/cover"):
        covered_files = set(os.listdir("dataset/cover"))

    with open(out, "w", encoding="utf-8") as f:
        f.write("FileName,Label\n")

        for p in list_images(inp):
            img = cv2.imread(p)
            lab = 0 if img is None else is_covered(img, scores)

            fname = os.path.basename(p)
            f.write(f"{fname},{lab}\n")

            y_pred.append(lab)
            y_true.append(1 if fname in covered_files else 0)

            # save sample images
            if len(y_pred) <= 6:
                show = img.copy()
                color = (0,255,0) if lab == 0 else (0,0,255)
                cv2.putText(show, f"Label: {lab}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imwrite(f"results/sample_{fname}", show)

    # =====================================================
    # PLOTS
    # =====================================================
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    plt.figure()
    plt.hist(scores, bins=20)
    plt.axvline(COVERAGE_THRESHOLD, linestyle="--")
    plt.title("Coverage Score Distribution")
    plt.savefig("results/score_distribution.png")
    plt.close()

    metrics = [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]

    plt.figure()
    plt.bar(["Accuracy","Precision","Recall","F1"], metrics)
    plt.ylim(0,1)
    plt.title("Performance Metrics")
    plt.savefig("results/performance_metrics.png")
    plt.close()

    print("✔ output.txt created")
    print("✔ Images and plots saved in /results")

# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output.txt")
    args = ap.parse_args()
    main(args.input, args.output)

