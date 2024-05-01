from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image, mask, color, alpha=0.5):
    # 이미지의 각 색상 채널(R, G, B)에 대해 반복
    for c in range(3):
        # image[:, :, c]는 이미지의 c번째 색상 채널을 의미
        # np.where을 사용하여 마스크가 1인 위치에서는 새로운 색상을 계산하여 적용
        image[:, :, c] = np.where(
            mask == 1,  # 마스크가 1인 조건
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,  # 색상 적용 조건
            image[:, :, c])  # 마스크가 1이 아닐 경우 원본 이미지 색상 유지
    # 수정된 이미지 반환
    return image




if __name__ == "__main__":
    
    '''results[0]는 list형으로 coco class의  0부터 79까지 class_id별로 80개의 array를 가짐.
    개별 array들은 각 클래스별로 5개의 값(좌표값과 class별로 confidence)을 가짐. 개별 class별로 여러개의 좌표를 가지면 여러개의 array가 생성됨.
    좌표는 좌상단(xmin, ymin), 우하단(xmax, ymax) 기준.
    개별 array의 shape는 (Detection된 object들의 수, 5(좌표와 confidence)) 임
    
    results[1]은 masking 정보를 가짐. coco class의  0부터 79까지 class_id 별로 80개의 list를 가짐. 개별 list는 개별 object의 mask 정보를 내부 원소로 가짐.
    개별 object의 mask 정보는 2차원 array로서 image의 height x width 형태를 가짐.
    '''

    # Specify the path to model config and checkpoint file
    config_file = 'configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    #img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
    img = 'demo/demo.jpg'
    img_arr  = cv2.imread(img)
    img_arr_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    results = inference_detector(model, img_arr_rgb)

    #1.show_result_pyplot 이용
    #show_result_pyplot(model, img, results)


    #2.만든 모듈 사용
    draw_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    masked_image = apply_mask(draw_img, results[1][2][0], (0, 255, 0), alpha=0.6) #results[1][2][0]의 경우 1값은 객체가 위치한 픽셀을 나타냄
    plt.figure(figsize=(12, 14))
    plt.imshow(masked_image)
    plt.axis('off')
    plt.show()
