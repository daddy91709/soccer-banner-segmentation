import torch, clip, cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Dizionario globale per memorizzare gli embeddings dei banner
embeddings = {}

def assign_content_ids(segmented, clip_model, clip_preprocess, prev_embeddings=None, threshold=0.25, device="cuda"):
    """
    Assegna content-ID ai banner segmentati usando CLIP.
    """
    if prev_embeddings is None:
        prev_embeddings = {}
    
    assigned = []
    next_content_id = max(prev_embeddings.keys(), default=0) + 1
    
    with torch.no_grad():
        for detection in segmented:
            crop = detection["crop"]
            
            # Converti crop in formato PIL per CLIP
            if crop is None or crop.size == 0:
                assigned.append({**detection, "content_id": -1})
                continue
            
            try:
                # CORREZIONE: Rimuovi i pixel neri prima di processare con CLIP
                crop_cleaned = remove_black_background(crop)
                
                if crop_cleaned is None:
                    assigned.append({**detection, "content_id": -1})
                    continue
                
                # Converti BGR->RGB
                if len(crop_cleaned.shape) == 3 and crop_cleaned.shape[2] == 3:
                    crop_rgb = cv2.cvtColor(crop_cleaned, cv2.COLOR_BGR2RGB)
                else:
                    crop_rgb = crop_cleaned
                
                # Converti in PIL Image
                pil_image = Image.fromarray(crop_rgb)
                
                # Preprocessing per CLIP
                image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)
                
                # Calcola embedding
                current_embedding = clip_model.encode_image(image_input)
                current_embedding = current_embedding / current_embedding.norm(dim=-1, keepdim=True)
                
                # Trova il content_id più simile
                best_similarity = -1
                best_content_id = None
                
                for content_id, stored_embedding in prev_embeddings.items():
                    stored_embedding = stored_embedding.to(device)
                    
                    similarity = F.cosine_similarity(
                        current_embedding, 
                        stored_embedding, 
                        dim=-1
                    ).item()
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_content_id = content_id
                
                # Assegna ID basato sulla similarità
                if best_similarity > threshold:
                    assigned_id = best_content_id
                    # Aggiorna embedding con media mobile
                    alpha = 0.7
                    prev_embeddings[assigned_id] = (
                        alpha * prev_embeddings[assigned_id].to(device) + 
                        (1 - alpha) * current_embedding
                    ).cpu()
                else:
                    # Nuovo contenuto
                    assigned_id = next_content_id
                    prev_embeddings[assigned_id] = current_embedding.cpu()
                    next_content_id += 1
                
                assigned.append({
                    **detection,
                    "content_id": assigned_id,
                    "similarity": best_similarity
                })
                
            except Exception as e:
                print(f"Errore CLIP processing per detection: {e}")
                assigned.append({**detection, "content_id": -1})
    
    return assigned, prev_embeddings

def remove_black_background(crop, black_threshold=10):
    """
    Rimuove lo sfondo nero dal crop, mantenendo solo i pixel colorati.
    
    Args:
        crop: immagine crop con sfondo nero
        black_threshold: soglia per considerare un pixel "nero"
    
    Returns:
        crop pulito senza sfondo nero, o None se troppo piccolo
    """
    # Trova pixel non neri
    if len(crop.shape) == 3:
        # Calcola la somma dei canali per identificare pixel non neri
        pixel_sum = np.sum(crop, axis=2)
        non_black_mask = pixel_sum > black_threshold
    else:
        non_black_mask = crop > black_threshold
    
    # Se ci sono troppo pochi pixel non neri, skippa
    non_black_pixels = np.sum(non_black_mask)
    total_pixels = crop.shape[0] * crop.shape[1]
    
    # Trova bounding box dei pixel non neri
    rows = np.any(non_black_mask, axis=1)
    cols = np.any(non_black_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Estrai solo la regione con contenuto
    if len(crop.shape) == 3:
        cropped_content = crop[rmin:rmax+1, cmin:cmax+1, :]
    else:
        cropped_content = crop[rmin:rmax+1, cmin:cmax+1]
    
    # Ridimensiona se troppo piccolo
    min_size = 32
    if cropped_content.shape[0] < min_size or cropped_content.shape[1] < min_size:
        if len(cropped_content.shape) == 3:
            cropped_content = cv2.resize(cropped_content, (min_size, min_size))
        else:
            cropped_content = cv2.resize(cropped_content, (min_size, min_size))
    
    return cropped_content