import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from algo import OptimizationEnv, hybrid_ga_sa

def handle_image_click(original_image, current_drawn_image, heat_sources, avoid_sources, valid_area, mode, evt: gr.SelectData):
    x, y = evt.index
    
    if mode == "Titik Panas":
        heat_sources.append({"x": x, "y": y, "intensity": 8000, "name": f"Source {len(heat_sources)+1}"})
    elif mode == "Tidak Disarankan":
        avoid_sources.append({"x": x, "y": y, "intensity": 6000, "name": f"Avoid {len(avoid_sources)+1}"})
    elif mode == "Area Valid":
        if len(valid_area) >= 2:
            valid_area.clear() # reset
        valid_area.append((x, y))
        
    img_copy = np.copy(original_image) if original_image is not None else current_drawn_image
        
    if img_copy is not None:
        # Draw valid area (Blue)
        if len(valid_area) == 2:
            x1, y1 = valid_area[0]
            x2, y2 = valid_area[1]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
        elif len(valid_area) == 1:
            cv2.circle(img_copy, valid_area[0], 10, (0, 0, 255), -1)
            
        # Draw avoid sources (Orange)
        for i, avd in enumerate(avoid_sources):
            cv2.circle(img_copy, (avd["x"], avd["y"]), 16, (255, 165, 0), -1)

        # Draw heat sources (Red)
        for i, src in enumerate(heat_sources):
            cv2.circle(img_copy, (src["x"], src["y"]), 20, (255, 0, 0), -1)
            
        return img_copy, heat_sources, avoid_sources, valid_area
    return current_drawn_image, heat_sources, avoid_sources, valid_area

def run_optimization(original_image, image_with_dots, heat_sources, avoid_sources, valid_area):
    if original_image is None:
        return None, "Harap unggah gambar terlebih dahulu.", avoid_sources
    if not heat_sources:
        return image_with_dots, "Harap klik pada gambar untuk menambahkan minimal 1 titik sumber panas (misal: CPU/GPU).", avoid_sources
        
    height, width, _ = original_image.shape
    
    # Estimated card size
    card_w, card_h = int(width * 0.12), int(height * 0.08)
    
    algo_bounds = None
    user_drawn_bounds = None
    
    if len(valid_area) == 2:
        x1, y1 = valid_area[0]
        x2, y2 = valid_area[1]
        
        raw_x_min, raw_x_max = min(x1, x2), max(x1, x2)
        raw_y_min, raw_y_max = min(y1, y2), max(y1, y2)
        
        # Draw bounds for visualization
        user_drawn_bounds = {
            'x_min': raw_x_min, 'x_max': raw_x_max,
            'y_min': raw_y_min, 'y_max': raw_y_max
        }
        
        # Calculate padding so the whole card fits inside the bounds
        calc_x_min = raw_x_min + card_w // 2
        calc_x_max = raw_x_max - card_w // 2
        if calc_x_min > calc_x_max:
            mid_x = (raw_x_min + raw_x_max) // 2
            calc_x_min, calc_x_max = mid_x, mid_x
            
        calc_y_min = raw_y_min + card_h // 2
        calc_y_max = raw_y_max - card_h // 2
        if calc_y_min > calc_y_max:
            mid_y = (raw_y_min + raw_y_max) // 2
            calc_y_min, calc_y_max = mid_y, mid_y
            
        algo_bounds = {
            'x_min': calc_x_min, 'x_max': calc_x_max,
            'y_min': calc_y_min, 'y_max': calc_y_max
        }
    else:
        # Default bounds: padding based on card size to avoid image edges
        algo_bounds = {
            'x_min': card_w // 2, 'x_max': width - 1 - card_w // 2,
            'y_min': card_h // 2, 'y_max': height - 1 - card_h // 2
        }
    
    env = OptimizationEnv(width, height, heat_sources, avoid_sources, algo_bounds)
    
    logs = []
    # Run Algorithm
    best_layout = hybrid_ga_sa(env, pop_size=80, generations=40, sa_iter=25, logs=logs)
    
    # --- Visualization ---
    # 1. Generate Heatmap
    scale = max(1, width // 100) # Ensure grid is manageable
    grid_w = width // scale
    grid_h = height // scale
    heatmap = np.zeros((grid_h, grid_w))
    
    for y in range(grid_h):
        for x in range(grid_w):
            heatmap[y, x] = env.calculate_heat(x * scale, y * scale)
            
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Normalize heatmap (0 to 1) for Colormap
    h_min, h_max = np.min(heatmap_resized), np.max(heatmap_resized)
    heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min + 1e-5)
    
    # Apply JET colormap
    heatmap_colored = plt.get_cmap('jet')(heatmap_norm)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Blend heatmap with original image
    alpha = 0.4
    result_img = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    
    # 2. Draw Result
    bx, by = best_layout['x'], best_layout['y']
    
    top_left = (max(0, bx - card_w//2), max(0, by - card_h//2))
    bottom_right = (min(width-1, bx + card_w//2), min(height-1, by + card_h//2))
    
    # Draw Green Bounding Box for Optimal Position
    cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 4)
    
    # Redraw avoid sources on the result (Orange)
    for avd in avoid_sources:
        cv2.circle(result_img, (avd['x'], avd['y']), 12, (255, 165, 0), -1)

    # Redraw heat sources on the result (Red)
    for src in heat_sources:
        cv2.circle(result_img, (src['x'], src['y']), 12, (255, 0, 0), -1)
        
    # Draw valid bounds (Blue)
    if user_drawn_bounds:
        cv2.rectangle(result_img, (user_drawn_bounds['x_min'], user_drawn_bounds['y_min']), (user_drawn_bounds['x_max'], user_drawn_bounds['y_max']), (0, 0, 255), 3)
        
    # Automatically save this found coordinate to avoid_sources for the NEXT iteration
    avoid_sources.append({"x": bx, "y": by, "intensity": 6000, "name": f"Hasil Sebelumnya {len(avoid_sources)+1}"})

    log_str = "\n".join(logs)
    msg = f"Sukses! Posisi Optimal Ditemukan di Piksel X: {bx}, Y: {by} | Skor Panas Minimum: {best_layout['heat']:.2f}\n\n=== LOG ITERASI ALGORITMA GA-SA ===\n{log_str}"
    return result_img, msg, avoid_sources

def handle_upload(img):
    return img, img, [], [], []

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Optimasi Letak Komponen ASUS TUF F15 (Hybrid GA & SA)")
    gr.Markdown("""### Cara Penggunaan:
    1. **Unggah** foto motherboard / layout laptop Anda.
    2. **Klik** pada gambar di sebelah kiri untuk menandai lokasi komponen yang menghasilkan panas.
    3. (Opsional) Pilih mode **Tidak Disarankan** untuk menandai area yang kurang ideal ditempati komponen (misal karena ada komponen kecil lain).
    4. (Opsional) Pilih mode **Area Valid**, lalu klik 2 titik (kiri atas dan kanan bawah) untuk membatasi area motherboard di mana WiFi card BISA diletakkan secara fisik.
    5. Setelah selesai menset parameter, klik tombol **Jalankan Optimasi GA-SA**. Hasil optimal yang ditemukan otomatis tersimpan sebagai area oranye (tidak disarankan) agar klik berikutnya mencari alternatif lain.
    """)
    
    # Store original image independently
    original_img_state = gr.State(None)
    heat_sources_state = gr.State([])
    avoid_sources_state = gr.State([])
    valid_area_state = gr.State([])
    
    with gr.Row():
        with gr.Column():
            click_mode = gr.Radio(["Titik Panas", "Tidak Disarankan", "Area Valid"], value="Titik Panas", label="Mode Objek (Klik Gambar)")
            input_image = gr.Image(label="1. Unggah & Klik Parameter Gambar", interactive=True)
            run_btn = gr.Button("🔍 Jalankan Optimasi GA-SA", variant="primary")
            clear_btn = gr.Button("🗑️ Reset Gambar & Titik")
            
        with gr.Column():
            output_image = gr.Image(label="2. Hasil Optimasi (Heatmap & Rekomendasi)")
            
            gr.Markdown("**Legenda Warna:**\n"
                        "\n🔴 **Merah**: Titik sumber panas (CPU/GPU)\n"
                        "\n🟠 **Oranye**: Titik tidak disarankan / hasil iterasi sebelumnya\n"
                        "\n🔵 **Biru**: Bounding box area valid tempat komponen menempel\n"
                        "\n🟢 **Hijau**: Posisi optimal penempatan (WIFI/SSD) yang disarankan")
                        
            output_text = gr.Textbox(label="Status / Log Iterasi Algoritma", interactive=False, lines=12, max_lines=15)

    # When image is uploaded, save a clean copy to original_img_state, update input_image, and clear sources
    input_image.upload(handle_upload, inputs=[input_image], outputs=[original_img_state, input_image, heat_sources_state, avoid_sources_state, valid_area_state])
    
    # Allow clearing
    clear_btn.click(lambda img: (img, [], [], []), inputs=[original_img_state], outputs=[input_image, heat_sources_state, avoid_sources_state, valid_area_state])
    
    # Handle clicks
    input_image.select(handle_image_click, inputs=[original_img_state, input_image, heat_sources_state, avoid_sources_state, valid_area_state, click_mode], outputs=[input_image, heat_sources_state, avoid_sources_state, valid_area_state])
    
    # Run optimization
    run_btn.click(run_optimization, inputs=[original_img_state, input_image, heat_sources_state, avoid_sources_state, valid_area_state], outputs=[output_image, output_text, avoid_sources_state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", theme=gr.themes.Soft())
