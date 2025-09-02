import cv2
import numpy as np
import os

# ==============================
# 1. CONFIGURATION
# ==============================
PAD_CAPACITY = {
    "pink": 3,
    "blue": 4,
    "grey": 2
}

CASUALTY_PRIORITY = {
    "star": 3,
    "triangle": 2,
    "square": 1
}

EMERGENCY_PRIORITY = {
    "red": 3,     # severe
    "yellow": 2,  # mild
    "green": 1    # safe
}


# ==============================
# 2. IMAGE SEGMENTATION
# ==============================
def segment_land_ocean(img):
    """Return a mask for land and ocean using color thresholding."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define thresholds (you will tune these!)
    ocean_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))  # blue-ish
    land_mask = cv2.inRange(hsv, (20, 40, 40), (85, 255, 255))   # green/brown

    segmented = img.copy()
    segmented[ocean_mask > 0] = (255, 0, 0)   # blue overlay
    segmented[land_mask > 0] = (0, 255, 0)    # green overlay

    return segmented, ocean_mask, land_mask


# ==============================
# 3. FEATURE DETECTION
# ==============================
def detect_casualties(img):
    casualties = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use edge detection instead of fixed threshold
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"[DEBUG] Found {len(contours)} potential casualties")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # skip tiny noise
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

        # --- Shape classification ---
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "square"
        elif len(approx) > 6:
            shape = "circle"  # treat circles as valid casualties
        else:
            shape = "unknown"

        # --- Color detection ---
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(hsv, mask=mask)
        h = mean_val[0]
        s = mean_val[1]
        v = mean_val[2]

        if (h < 15 or h > 165) and s > 50:
            color = "red"
        elif 15 <= h < 40:
            color = "yellow"
        elif 40 <= h < 90:
            color = "green"
        else:
            color = "unknown"

        if shape in CASUALTY_PRIORITY and color in EMERGENCY_PRIORITY:
            priority = CASUALTY_PRIORITY[shape] * EMERGENCY_PRIORITY[color]
            casualties.append({
                "shape": shape,
                "color": color,
                "coords": (cx, cy),
                "priority": priority
            })

    print(f"[DEBUG] Valid casualties detected: {len(casualties)}")
    return casualties



def detect_rescue_pads(img):
    pads = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Relaxed Hough parameters
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=20, minRadius=10, maxRadius=200
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        print(f"[DEBUG] Found {len(circles[0])} rescue pads")

        for i, (x, y, r) in enumerate(circles[0, :]):
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_val = cv2.mean(hsv, mask=mask)
            h = mean_val[0]
            s = mean_val[1]
            v = mean_val[2]

            # --- Color classification ---
            if (h < 15 or h > 165) and s > 50:
                color = "pink"
            elif 90 < h < 130:
                color = "blue"
            elif v < 80:  # low brightness → grey
                color = "grey"
            else:
                color = f"pad{i}"

            pads.setdefault(color, []).append((x, y))

    else:
        print("[DEBUG] No rescue pads detected")

    return pads





# ==============================
# 4. ASSIGN CASUALTIES
# ==============================

def visualize(img, casualties, pads, assignments):
    out = img.copy()
    # draw casualties
    for c in casualties:
        cv2.circle(out, c["coords"], 10, (0,0,255), -1)
        cv2.putText(out, f"{c['shape']}-{c['color']}", c["coords"],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # draw pads
    for pad, coords_list in pads.items():
       for coords in coords_list:
        cv2.circle(out, coords, 15, (255,255,0), 2)

        cv2.putText(out, pad, (coords[0]+10, coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    # draw assignments (lines)
    for pad, cas_list in assignments.items():
        for c in cas_list:
            cv2.line(out, c["coords"], pads[pad], (0,255,0), 2)
    return out


def assign_casualties(casualties, pads):
    """
    Assign casualties to nearest/best pad based on priority & distance.
    """
    pad_assignments = {pad: [] for pad in pads.keys()}
    pad_scores = {pad: 0 for pad in pads.keys()}

    for casualty in casualties:
        best_pad = None
        best_score = -1

        for pad, coords in pads.items():
            dist = np.linalg.norm(np.array(casualty["coords"]) - np.array(coords))
            score = casualty["priority"] / (dist + 1e-6)  # avoid divide by zero

            # Check capacity
            if len(pad_assignments[pad]) < PAD_CAPACITY[pad]:
                if score > best_score:
                    best_score = score
                    best_pad = pad

        if best_pad:
            pad_assignments[best_pad].append(casualty)
            pad_scores[best_pad] += casualty["priority"]

    return pad_assignments, pad_scores


# ==============================
# 5. MAIN PIPELINE
# ==============================
def process_image(path):
    img = cv2.imread(path)

    segmented, ocean_mask, land_mask = segment_land_ocean(img)
    casualties = detect_casualties(img)
    pads = detect_rescue_pads(img)

    assignments, scores = assign_casualties(casualties, pads)

    total_priority = sum(scores.values())
    avg_priority = total_priority / len(casualties) if casualties else 0

    return {
        "segmented": segmented,
        "assignments": assignments,
        "scores": scores,
        "avg_priority": avg_priority
    }


if __name__ == "__main__":
    folder = "C:\\Users\\Rishabh Goyal\\Desktop\\projects\\images"
    results = []

    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            res = process_image(os.path.join(folder, filename))
            results.append((filename, res["avg_priority"]))

            # Save segmented image
            
            os.makedirs("output", exist_ok=True)


            # Print assignment summary
            print(f"{filename}: Avg Rescue Ratio = {res['avg_priority']:.2f}")

    # Sort by rescue ratio
    results.sort(key=lambda x: x[1], reverse=True)
    print("Images by Rescue Ratio:", [r[0] for r in results])