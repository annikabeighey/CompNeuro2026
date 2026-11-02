import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

def batch_process_identity(source_dir, inverted_dir, output_csv_path):
    """
    Scans source and inverted directories, matches image pairs by filename,
    calculates ArcFace cosine similarity, and saves results to a CSV file.
    """
    # 1. Initialize the official InsightFace App
    print("Initializing ArcFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 2. Gather image files from the source directory
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(valid_extensions)]
    
    results = []
    
    print(f"Found {len(source_files)} source images. Processing pairs...")
    
    # 3. Process pairs
    for filename in source_files:
        source_path = os.path.join(source_dir, filename)
        inverted_path = os.path.join(inverted_dir, filename)
        
        # Check if matching inverted image exists
        if not os.path.exists(inverted_path):
            print(f"Warning: Matching inverted image not found for {filename}. Skipping.")
            continue
            
        try:
            # Load images
            img_src = cv2.imread(source_path)
            img_inv = cv2.imread(inverted_path)
            
            # Extract features
            faces_src = app.get(img_src)
            faces_inv = app.get(img_inv)
            
            if len(faces_src) == 0:
                print(f"Skipping {filename}: No face detected in source image.")
                status = "Error: No source face"
                score = np.nan
            elif len(faces_inv) == 0:
                print(f"Skipping {filename}: No face detected in inverted image.")
                status = "Error: No inverted face"
                score = np.nan
            else:
                # Get 512-D L2-normalized embeddings for the first/primary face detected
                emb_src = faces_src[0].normed_embedding
                emb_inv = faces_inv[0].normed_embedding
                
                # Since they are pre-normalized, dot product is exactly cosine similarity
                score = float(np.dot(emb_src, emb_inv))
                status = "Success"
                
            results.append({
                "Filename": filename,
                "Source_Path": source_path,
                "Inverted_Path": inverted_path,
                "Cosine_Similarity": score,
                "Status": status
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            results.append({
                "Filename": filename,
                "Source_Path": source_path,
                "Inverted_Path": inverted_path,
                "Cosine_Similarity": np.nan,
                "Status": f"Error: {str(e)}"
            })

    # 4. Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nProcessing complete! Results saved to {output_csv_path}")
    
    # Print high-level summary stats
    successful_runs = df[df["Status"] == "Success"]
    if not successful_runs.empty:
        print(f"Average Identity Preservation Score: {successful_runs['Cosine_Similarity'].mean():.4f}")
    
    return df

# Example Usage setup:
# batch_process_identity(
#     source_dir="path/to/source_faces", 
#     inverted_dir="path/to/inverted_faces", 
#     output_csv_path="identity_evaluation_results.csv"
# )
