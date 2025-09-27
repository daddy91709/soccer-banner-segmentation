# utils.py
import os, glob, shutil, random, cv2, re
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def find_matching_image_for_mask(mask_path, images_dir, image_exts=(".jpg",".jpeg",".png",".bmp")):
    """
    Trova l'immagine corrispondente a una maschera.
    Esempio: mask0.png -> frame0.jpg
    Usa pattern numerico per fare il match.
    """
    mask_stem = os.path.splitext(os.path.basename(mask_path))[0]  # "mask0"
    # estrai prima sequenza di cifre
    m = re.search(r'(\d+)', mask_stem)
    if m:
        num = m.group(1).lstrip("0") or "0"  # rimuove leading zeros
        # prova forma "frame{num}" con varie estensioni
        for ext in image_exts:
            cand = os.path.join(images_dir, f"frame{num}{ext}")
            if os.path.exists(cand):
                return cand
        # prova anche con leading zeros (es. frame001.jpg)
        for ext in image_exts:
            for z in range(1,4):  # fino a 3 zeri davanti
                cand = os.path.join(images_dir, f"frame{num.zfill(z+len(num))}{ext}")
                if os.path.exists(cand):
                    return cand
        # prova file che contengono il numero nel nome
        for p in glob.glob(os.path.join(images_dir, "*.*")):
            if num in os.path.basename(p):
                return p
    # ultima risorsa: stesso nome con estensione diversa
    for ext in image_exts:
        cand = os.path.join(images_dir, mask_stem + ext)
        if os.path.exists(cand):
            return cand
    return None

# --- YOLO ---

def masks_to_yolo(
    images_dir,
    masks_dir,
    out_dir="dataset",
    min_area=150,
    seed=42,
    split=(0.8, 0.1, 0.1),
    class_id=0
):
    """
    Converte maschere binarie in annotazioni YOLOv8 (bounding boxes).
    - images_dir: cartella con immagini originali
    - masks_dir: cartella con maschere (es. mask0.png)
    - out_dir: output dataset YOLO (con images/, labels/, football.yaml)
    - min_area: area minima per considerare una bbox (in pixel)
    - seed: seme random per lo split train/val/test
    - split: proporzioni (train, val, test)
    - class_id: id numerico della classe (default 0 -> 'adboard')
    """
    os.makedirs(out_dir, exist_ok=True)
    for s in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)

    mask_files = glob.glob(os.path.join(masks_dir, "*.*"))
    pairs = []
    for m in mask_files:
        matched_img = find_matching_image_for_mask(m, images_dir)
        if matched_img:
            pairs.append((matched_img, m))
        else:
            print(f"[ATTENZIONE] Nessuna immagine trovata per {os.path.basename(m)}")

    if len(pairs) == 0:
        raise RuntimeError("Nessuna coppia immagine/maschera trovata. Controlla i path.")

    print(f"Trovate {len(pairs)} coppie immagine/maschera. Creo annotazioni YOLO...")

    records = []
    for img_path, mask_path in tqdm(pairs):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for c in contours:
            x,y,ww,hh = cv2.boundingRect(c)
            if ww*hh < min_area:
                continue
            cx = (x + ww/2) / w
            cy = (y + hh/2) / h
            nw = ww / w
            nh = hh / h
            bboxes.append((class_id, cx, cy, nw, nh))
        if bboxes:
            records.append((img_path, bboxes))

    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train = records[:n_train]
    val = records[n_train:n_train+n_val]
    test = records[n_train+n_val:]

    def save_split(split_records, split_name):
        for img_path, bboxes in split_records:
            img_name = os.path.basename(img_path)
            stem = os.path.splitext(img_name)[0]
            dst_img = os.path.join(out_dir, "images", split_name, img_name)
            dst_label = os.path.join(out_dir, "labels", split_name, stem + ".txt")
            shutil.copyfile(img_path, dst_img)
            with open(dst_label, "w") as f:
                for (cls, cx, cy, nw, nh) in bboxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    save_split(train, "train")
    save_split(val, "val")
    save_split(test, "test")

    yaml_path = os.path.join(out_dir, "football.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 1\n")
        f.write("names: ['adboard']\n")

    print("Conversione completata ✅")
    print("Dataset pronto in:", out_dir)
    print("Config YAML:", yaml_path)



def prepare_crops(results):
    frames = []
    boxes_lists = []
    
    for result in results:
        # Estrai frame - usa direttamente result.orig_img
        frame = result.orig_img.copy()
        frames.append(frame)
        
        # Estrai boxes
        frame_boxes = []
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for i in range(len(result.boxes.xyxy)):
                bbox_dict = {
                    "xyxy": result.boxes.xyxy[i],
                    "conf": result.boxes.conf[i] if result.boxes.conf is not None else 1.0,
                    "cls": result.boxes.cls[i] if result.boxes.cls is not None else 0,
                    "id": result.boxes.id[i] if result.boxes.id is not None else i
                }
                frame_boxes.append(bbox_dict)
        
        boxes_lists.append(frame_boxes)
    
    return frames, boxes_lists

# --- SAM2 ---

def run_sam_on_detections(frames, detections, sam_predictor):
    """
    Applica SAM2 alle bounding box fornite da YOLO+Tracker e restituisce i crop segmentati.
    """
    import numpy as np
    segmented = []
    
    for frame_idx, (frame, frame_detections) in enumerate(zip(frames, detections)):
        frame_segmented = []
        
        if len(frame_detections) == 0:
            segmented.append(frame_segmented)
            continue
            
        # Imposta l'immagine per SAM2
        sam_predictor.set_image(frame)
        
        for detection in frame_detections:
            try:
                # Estrai bbox
                xyxy = detection["xyxy"]
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                xyxy = xyxy.flatten()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Verifica che la bbox sia valida
                if x2 <= x1 or y2 <= y1:
                    print(f"Bbox non valida frame {frame_idx}: {x1},{y1},{x2},{y2}")
                    continue
                
                # Usa la bbox come prompt per SAM2
                input_box = np.array([x1, y1, x2, y2])
                
                # Genera maschera con SAM2
                masks, scores, logits = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Prendi la migliore maschera e assicurati sia booleana
                mask = masks[0]  # shape: (H, W)
                if mask.dtype != bool:
                    mask = mask.astype(bool)
                
                # Crea crop segmentato
                segmented_crop = apply_mask_to_crop(frame, mask, x1, y1, x2, y2)
                
                # Aggiungi ai risultati
                frame_segmented.append({
                    "xyxy": xyxy,
                    "mask": mask,
                    "crop": segmented_crop,
                    "track_id": detection.get("id", -1),  # Cambiato da track_id a id
                    "conf": detection.get("conf", 0.0)
                })
                
            except Exception as e:
                print(f"Errore SAM2 frame {frame_idx}: {e}")
                # Fallback: crop rettangolare normale
                try:
                    normal_crop = frame[y1:y2, x1:x2]
                    frame_segmented.append({
                        "xyxy": xyxy,
                        "mask": None,
                        "crop": normal_crop,
                        "track_id": detection.get("id", -1),
                        "conf": detection.get("conf", 0.0)
                    })
                except:
                    # Skip questa detection se anche il fallback fallisce
                    continue
        
        segmented.append(frame_segmented)
    
    return segmented

def apply_mask_to_crop(frame, mask, x1, y1, x2, y2):
    """
    Applica la maschera al crop e restituisce l'immagine segmentata con sfondo trasparente.
    """
    # Crop della regione
    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    
    # Assicurati che la maschera sia booleana
    if crop_mask.dtype != bool:
        crop_mask = crop_mask.astype(bool)
    
    # Applica maschera (pixel non-mask diventano neri)
    crop[~crop_mask] = [0, 0, 0]  # sfondo nero
    
    return crop

def load_finetuned_sam2(checkpoint_path, device="cuda"):
    """
    Carica un modello SAM2 fine-tuned.
    """
    # Carica il checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get('model_name', 'facebook/sam2.1-hiera-tiny')
    
    print(f"Caricando modello base: {model_name}")
    
    # 🎯 CARICA IL MODELLO BASE con from_pretrained
    sam_predictor = SAM2ImagePredictor.from_pretrained(model_name)
    
    # 🎯 CARICA I PESI FINE-TUNED
    sam_predictor.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Modello fine-tuned caricato da {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return sam_predictor