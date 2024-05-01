from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_detected_img(model, img_array,  score_threshold=0.3, draw_box=True, is_print=True):

    labels_to_names_seq =  {0:'balloon'}
    
    colors = list(
        [[0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250]] )
    
    # 인자로 들어온 image_array를 복사. 
    draw_img = img_array.copy()
    bbox_color=(0, 255, 0)
    text_color=(0, 0, 255)

    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음. 
    results = inference_detector(model, img_array)
    bbox_results = results[0]
    seg_results = results[1]

    # results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화 
    # results 리스트의 위치 index가  Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐. 
    for result_ind, bbox_result in enumerate(bbox_results):
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행. 
        if len(bbox_result) == 0:
            continue
        
        mask_array_list = seg_results[result_ind]
        
        # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
        for i in range(len(bbox_result)):
            # 좌상단, 우하단 좌표 추출. 
            if bbox_result[i, 4] > score_threshold: #score가 threshold보다 높으면
                left = int(bbox_result[i, 0])
                top = int(bbox_result[i, 1])
                right = int(bbox_result[i, 2])
                bottom = int(bbox_result[i, 3])
                caption = "{}: {:.4f}".format(labels_to_names_seq[result_ind], bbox_result[i, 4])
            if draw_box:
                cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
                cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            # masking 시각화 적용. class_mask_array는 image 크기 shape의  True/False값을 가지는 2차원 array
            class_mask_array = mask_array_list[i]
            # 원본 image array에서 mask가 True인 영역만 별도 추출. 
            masked_roi = draw_img[class_mask_array]
            #color를 임의 지정
            #color_index = np.random.randint(0, len(colors)-1)
            # color를 class별로 지정
            color_index = result_ind % len(colors)
            color = colors[color_index]

            # apply_mask()함수를 적용시 수행 시간이 상대적으로 오래 걸림. 
            #draw_img = apply_mask(draw_img, class_mask_array, color, alpha=0.4)
            # 원본 이미지의 masking 될 영역에 mask를 특정 투명 컬러로 적용
            draw_img[class_mask_array] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.6 * masked_roi).astype(np.uint8) #색상에 30%씩 원본 이미지에 60%를 유지
            
            if is_print:
                print(caption)
    
    return draw_img

# 원본 이미지를 Gray scale로 변환하고, 컬러 기반 instance segmentation을 적용하는 함수
def get_detected_img_n_gray(model, img_array,  score_threshold=0.3, draw_box=True, is_print=True):
    labels_to_names_seq =  {0:'balloon'}
    
    colors = list(
        [[0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250]] )
    
    # 인자로 들어온 image_array를 복사. 
    draw_img = img_array.copy()
    bbox_color=(0, 255, 0)
    text_color=(0, 0, 255)

    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음. 
    results = inference_detector(model, img_array)
    bbox_results = results[0]
    seg_results = results[1]
    # 원본 이미지를 Grayscale로 변환. BGR2GRAY적용시 2차원 array로 변환되므로 다시 GRAY2BGR로 변환하면 3차원이지만, 여전히 Grayscale임. 
    draw_img_gray = cv2.cvtColor(cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화 
    # results 리스트의 위치 index가  Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐. 
    for result_ind, bbox_result in enumerate(bbox_results):
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행. 
        if len(bbox_result) == 0:
            continue
        
        mask_array_list = seg_results[result_ind]
        
        # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
        for i in range(len(bbox_result)):
            # 좌상단, 우하단 좌표 추출. 
            if bbox_result[i, 4] > score_threshold: #score가 threshold보다 높으면
                left = int(bbox_result[i, 0])
                top = int(bbox_result[i, 1])
                right = int(bbox_result[i, 2])
                bottom = int(bbox_result[i, 3])
                caption = "{}: {:.4f}".format(labels_to_names_seq[result_ind], bbox_result[i, 4])
            if draw_box:
                cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
                cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            # masking 시각화 적용. class_mask_array는 image 크기 shape의  True/False값을 가지는 2차원 array
            class_mask_array = mask_array_list[i]
            # 원본 image array에서 mask가 True인 영역만 별도 추출. 
            masked_roi = draw_img[class_mask_array]
            #color를 임의 지정
            #color_index = np.random.randint(0, len(colors)-1)
            # color를 class별로 지정
            color_index = result_ind % len(colors)
            color = colors[color_index]

            # apply_mask()함수를 적용시 수행 시간이 상대적으로 오래 걸림. 
            #draw_img = apply_mask(draw_img, class_mask_array, color, alpha=0.4)
            # 원본 이미지의 masking 될 영역에 mask를 특정 투명 컬러로 적용
            draw_img_gray[class_mask_array] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.6 * masked_roi).astype(np.uint8) #색상에 30%씩 원본 이미지에 60%를 유지
            
            if is_print:
                print(caption)
    
    return draw_img_gray


if __name__ == "__main__":
    # Config 설정하고 Pretrained 모델 다운로드
    config_file = 'configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'
    checkpoint_file = 'sg_weights/epoch_12.pth'

    cfg = Config.fromfile(config_file)
    
    # dataset에 대한 환경 파라미터 수정.
    cfg.data.val.type = 'BalloonDataset'
    cfg.data.val.data_root = 'balloon'
    cfg.data.val.ann_file = 'val_coco.json'
    cfg.data.val.img_prefix = 'val'

    # class의 갯수 수정. balloon 단일 class
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1
    
    # pretrained 모델 로드 경로 설정
    cfg.load_from = checkpoint_file

    # ConfigDict' object has no attribute 'device 오류 대응
    cfg.device = 'cuda'

    # 모델 초기화 및 디바이스 설정
    model = init_detector(cfg, checkpoint_file, device=cfg.device)

    # 추론을 위한 이미지 경로 설정
    img = 'balloon/val/16335852991_f55de7958d_k.jpg'

    # 추론 수행
    results = inference_detector(model, img)
    #results[0] = bbox 정보 , results[1] = segmentation정보
    # bbox_results = results[0]
    # seg_results = results[1]

    # 1. show_result_pyplot 사용
    #show_result_pyplot(model, img, results, score_thr=0.3)

    #2. 만든 함수 사용
    img_arr = cv2.imread(img)
    
    #rgb
    detected_img = get_detected_img(model, img_arr,  score_threshold=0.3, draw_box=True, is_print=True)
    #gray
    #detected_img = get_detected_img_n_gray(model, img_arr,  score_threshold=0.3, draw_box=False, is_print=True)
    # detect 입력된 이미지는 bgr임. 이를 최종 출력시 rgb로 변환 
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 14))
    plt.imshow(detected_img)
    plt.show()