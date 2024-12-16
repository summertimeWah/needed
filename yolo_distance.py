from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supabase import create_client, Client
import numpy as np
import cv2
import os
import base64
import json
import random
import logging
from datetime import datetime
import torch
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from models.experimental import attempt_load
from PIL import Image

from utils.plots import plot_one_box

from concurrent.futures import ThreadPoolExecutor
import asyncio

# 初始化 FastAPI 應用程式
app = FastAPI()

# 這些變數控制著應用程序的基本運行
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

frame_count = 0
distance = 0
width = 0
height = 0

# 設置 Supabase Client
SUPABASE_URL = "https://oxskmydkkwzllyxnbcny.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94c2tteWRra3d6bGx5eG5iY255Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyNzg0MjEyMywiZXhwIjoyMDQzNDE4MTIzfQ.tDqV4zXnhChIlDN0EUHJaPSogFjtIWTBLDxuufX1hDs"  # 替換為您的 API 金鑰
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

response = supabase.table("video").select("*").execute()

#user_id = "61666aaa-2d3a-4898-90fa-1d23bda31fc2"
user_id = "yes"
userId = ""

# YOLO 初始化
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
weights = 'best.pt'  
model = attempt_load(weights, map_location=device)
model = model.to(device)
model.eval()

'''
# 初始化 Metric3D 模型
metric3d_cfg_file = 'mono/configs/HourglassDecoder/vit.raft5.small.py'
metric3d_ckpt_file = 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth'
metric3d_cfg = Config.fromfile(metric3d_cfg_file)
metric3d_model = get_configured_monodepth_model(metric3d_cfg)
metric3d_model.load_state_dict(
    torch.hub.load_state_dict_from_url(metric3d_ckpt_file)['model_state_dict'],
    strict=False,
)
metric3d_model.cuda().eval()'''
metric3d_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
metric3d_model = metric3d_model.to(device)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# 内参和输入大小
intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
input_size = (616, 1064)

def prepare_image_for_metric3d(im0, intrinsic, input_size):
    rgb_origin = im0[:, :, ::-1]  # BGR to RGB
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    mean = torch.tensor([123.675, 116.28, 103.53]).float().view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).float().view(3, 1, 1)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = (rgb - mean) / std
    rgb = rgb[None, :, :, :].cuda()
    return rgb, [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half], intrinsic

def postprocess_depth(pred_depth, pad_info, im0_shape, intrinsic):
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]:pred_depth.shape[0] - pad_info[1], pad_info[2]:pred_depth.shape[1] - pad_info[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], im0_shape[:2], mode='bilinear').squeeze()
    canonical_to_real_scale = intrinsic[0] / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale
    pred_depth = torch.clamp(pred_depth, 0, 300)
    return pred_depth


logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

executor = ThreadPoolExecutor(max_workers=4)  # 限制併行工作數量


async def process_frame_async(frame, websocket):
    try:
        # 使用 YOLO 和 Metric3D 運算
        distance = await asyncio.to_thread(detect_and_compute_depth, frame) # type: ignore

        # 如果需要保存處理後的圖片
        frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)

        # 回傳運算結果
        await websocket.send_text(json.dumps({"status": "frame processed", "distance": distance}))
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")


# WebSocket 端點
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global frame_count, width, height
    await websocket.accept()

    logger = logging.getLogger('uvicorn.error')
    logger.setLevel(logging.DEBUG)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "recording_status":
                is_recording = message.get('isRecording')
                print(f"Recording status: {is_recording}")
                userId = message.get('userid')
                print(type(userId))
                if not is_recording:
                    print("Finalizing video and saving to database...")
                    await finalize_video(logger)
                else:
                    global user_id
                    user_id = userId

            elif message["type"] == "frame":
                print("Frame received and processed")

                base64_image = message.get('frame')
                width = message.get('width')
                height = message.get('height')

                if base64_image and width and height:
                    frame_data = base64.b64decode(base64_image)
                    frame = process_nv21_frame(frame_data, width, height)

                    # 使用 YOLO 模型進行檢測
                    distance = int(detect_and_compute_depth(frame))

                    # 保存處理後的圖片
                    frame_path = os.path.join(FRAME_DIR, f"frame_{frame_count:05d}.png")
                    cv2.imwrite(frame_path, frame)

                    #asyncio.create_task(process_frame_async(frame, websocket))

                    frame_count += 1

                    await websocket.send_text(json.dumps({"status": "frame received", "distance": distance}))
    except WebSocketDisconnect:
        logger.debug("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        logger.debug("Video creation finalized")

# 處理 NV21 格式的影像
def process_nv21_frame(nv21_data, width, height):
    expected_size = width * height * 3 // 2
    if len(nv21_data) != expected_size:
        raise ValueError(f"Invalid frame size: expected {expected_size}, got {len(nv21_data)}")

    yuv_image = np.frombuffer(nv21_data, np.uint8).reshape((height * 3 // 2, width))
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    
    # 顯示影像以驗證處理
    cv2.imshow("Frame", bgr_image)
    cv2.waitKey(1)

    return bgr_image

def detect_and_compute_depth(frame):
    # YOLO detec
    img = letterbox(frame, 640, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred =model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)

    # Metric3D
    rgb, pad_info, intrinsic_rescaled = prepare_image_for_metric3d(frame, intrinsic, input_size)
    rgb = rgb.to(device)
    with torch.no_grad():
        pred_depth, _, _ = metric3d_model.inference({'input': rgb})
    pred_depth = postprocess_depth(pred_depth, pad_info, frame.shape, intrinsic_rescaled)
    depth_map = pred_depth.cpu().numpy()

    # 找出檢測框中的距離
    all_depths = []
    detection_depth = 100000000
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                region_depth = depth_map[y1:y2, x1:x2]
                logger.debug(region_depth)
                detection_depth = region_depth.mean() if depth_map is not None else 100000000
                #logger.debug(detection_depth)
                all_depths.append(detection_depth)


    #min_distance = min(all_depths) if all_depths is not None else 100000000

    '''                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                detection_depth = depth_map[y1:y2, x1:x2].mean() if depth_map is not None else 0
                #label = f'{names[int(cls)]} {conf:.2f} Depth: {detection_depth:.2f}m'
                label = f'{names[int(cls)]} {conf:.2f} Depth: {detection_depth:.2f}m'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=4)
    
    save_path = 'C:/GitHub/testFastAPI/yolov7/detection.jpg'
    cv2.imwrite(save_path, frame)
    print(f" The image with the result is saved in: {save_path}")'''
    print(detection_depth)
    return detection_depth * 10
# 最後將所有幀合併成影片並上傳到 Supabase
async def finalize_video(logger):
    global frame_count, width, height

    # 保存影片
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('output_video.mp4', fourcc, 16.0, (width, height))

    for i in range(frame_count):
        frame_path = os.path.join(FRAME_DIR, f"frame_{i:05d}.png")
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()

    # 清理臨時圖片
    for i in range(frame_count):
        os.remove(os.path.join(FRAME_DIR, f"frame_{i:05d}.png"))

    frame_count = 0
    output_video_path = "output_video.mp4"

    try:
        # 將影片上傳到 Supabase
        bucket_name = "video"
        video_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # 讀取影片檔案並上傳
        with open(output_video_path, 'rb') as video_file:
            video_data = video_file.read()

        response = supabase.storage.from_("video").upload(
            file=video_data,
            path=video_name,
            file_options={"content-type": "video/mp4"}
        )

        # 檢查是否上傳成功
        if response.status_code != 200:
            logger.error(f"Error uploading video to Supabase: {response.json()}")
            return
        
        # 生成影片 URL 並插入資料表
        video_url = f"https://{SUPABASE_URL.replace('https://', '')}/storage/v1/object/public/{bucket_name}/{video_name}"

        video_record = {
            "uid": user_id,
            "url": video_url,
            "videoName": video_name,
            "created_at": datetime.utcnow().isoformat()
        }

        print(user_id)

        insert_response = supabase.table('video').insert(video_record).execute()

        if insert_response.get('error'):
            logger.error(f"Error inserting video record into Supabase: {insert_response.json()}")
        else:
            logger.debug("Video record inserted successfully.")

    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")

if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    image_path = cv2.imread('test.png')  # This will be automatically read in BGR format

    detect_and_compute_depth(image_path)
