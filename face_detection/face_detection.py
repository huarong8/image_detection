import sys
import cv2
import os
import pathlib
import traceback

fix_frame = 4
haar = "/Users/haiboz/practice/opencv-4.5.3/data/haarcascades/haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(haar)

face_coord = ()
w = 1080
h = 610
face_resize_rate = 1.9
resize_rate = 1.5
fixed_move = 240

def video_to_clips(video_file, output_folder, resize=1):
    os.makedirs(output_folder, exist_ok=True)
    video_cap = cv2.VideoCapture(str(video_file))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    vid_name = os.path.splitext(os.path.basename(str(video_file)))[0]
    clip_name = os.path.join(output_folder, '%s_clip_%%05d.mp4' % vid_name)
    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(video_length)

    init = True
    file_name = str(video_file.name)
    shot_frame_list = video_shot_frames[file_name]

    clip_cap_dict = {}
    p = 0
    while video_cap.isOpened():
        # Get the next video frame
        _, frame = video_cap.read()
        #print(video_file, frame.shape)
        #sys.exit(1)

        # Resize the frame
        if resize != 1 and frame is not None:
            frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
        if frame is None:
            print('end..... There was a problem processing frame %d' %  p)
            for clip_cap_item in clip_cap_dict.values():
                clip_cap_item.release()
            break
        # 裁剪底部字幕
        frame = frame[0:-100, :, :]

        if init:
            i = 0
            frame_size = frame.shape
            for frame_index in shot_frame_list[1:]:
                #clip_cap = cv2.VideoWriter(clip_name % i,fourcc,fps,(frame_size[1], frame_size[0]))
                clip_cap = cv2.VideoWriter(clip_name % i,fourcc,fps,(w, h))
                clip_cap_dict[frame_index] = clip_cap
                i+=1
            # face
            face_coord = detect_face(frame)
            init = False

        current_clip_index = shot_frame_list[1]
        for index in shot_frame_list[1:]:
            if p <= index:
                current_clip_index = index
                break

        #计算每个视频的第一帧是否有人脸
        for index in shot_frame_list[1:]:
            if p == index+1:
                face_coord = detect_face(frame)
                break

        ##跳过最后5帧
        if p+fix_frame <= current_clip_index:
            #print(p, current_clip_index)
            #对frmae裁剪 crop
            crop_frame = []
            if len(face_coord) > 0:
                crop_frame = format_img(frame, True)
            else:
                crop_frame = format_img(frame, False)
            #print(current_clip_index, crop_frame.shape)
            #crop_frame = frame
            #print("shot_index:", current_clip_index, p)
            clip_cap_dict[current_clip_index].write(crop_frame)
        p+=1

def format_img(frame, is_face):
    rate = resize_rate
    if is_face:
        rate = face_resize_rate

    x0 = int((rate - 1) * w/2)
    x1 = x0 + w
    y0 = int((rate - 1) * h/2) + fixed_move
    y1 = y0 + h
    #x0 = 0
    #x1 = 100
    #y0 = 0
    #y1 = 100

    frame = cv2.resize(frame, (w, h))
    frame = cv2.resize(frame, (0, 0), fx=rate, fy=rate)
    #print(frame.shape)
    #print("======")
    #print("crop_coord", x0, x1, y0, y1)
    #print("frame_size:", frame.shape)
    frame = frame[y0:y1, x0:x1,:]
    return frame

def detect_face(img):
    coord = ()
    #裁剪frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
    # OpenCV人脸识别分类器
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        faceRect = faceRects[0]
        x, y, w, h = faceRect
        coord = (x, y, w, h)
        #for faceRect in faceRects:  # 单独框出每一张人脸
        #    x, y, w, h = faceRect
        #    print(x,y, w,h)
        #    # 框出人脸
        #    cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
    #TODO 人脸在图像中心点
    if len(coord)>0:
        x, y, w, h = coord
        ih, iw, _ = img.shape
        ch = int(ih/2)
        cw = int(iw/2)
        if ch > y and ch < (y+h) and cw > x and cw < (x+w):
            coord = (x, y, w, h)
        else:
            coord = ()
        #center =
    return coord


def get_shot_frames(f):
    result = {}
    fp = open(f, "r")
    for line in fp:
        job_id, video_file, shot_frames = line.strip().split("\t")
        file_name = video_file.replace("shot_detector/", "", 1)
        result[file_name] = eval(shot_frames)
    return result

if __name__ == "__main__":
    video_shot_file= "reult.log"
    video_shot_frames = get_shot_frames(video_shot_file)
    #print(video_shot_frames)
    #sys.exit(1)

    out_dir = "./out_1"
    input_dir = "./shot_detector"
    p = pathlib.Path(input_dir)
    file_list = p.glob("*.mp4")
    i = 0
    for f in file_list:
        try:
            video_to_clips(f, out_dir)
        except:
            traceback.print_exc()
        finally:
            i+=1


