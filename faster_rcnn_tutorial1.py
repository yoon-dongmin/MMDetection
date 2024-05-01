from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# model과 원본 이미지 array, filtering할 기준 class confidence score를 인자로 가지는 inference 시각화용 함수 생성.
def get_detected_img(model, img_array,  score_threshold=0.3, is_print=True):

    labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' }
    
    # 인자로 들어온 image_array를 복사.
    draw_img = img_array.copy()
    bbox_color=(0, 255, 0)
    text_color=(0, 0, 255)

    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음.
    # results는 80개의 2차원 array(shape=(오브젝트갯수, 5))를 가지는 list.
    results = inference_detector(model, img_array)

    # 80개의 array원소를 가지는 results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화
    # results 리스트의 위치 index가 바로 COCO 매핑된 Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐.
    for result_ind, result in enumerate(results):
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값이 없으므로 다음 loop로 진행.
        if len(result) == 0:
            continue

        # 2차원 array에서 5번째 컬럼에 해당하는 값이 score threshold이며 이 값이 함수 인자로 들어온 score_threshold 보다 낮은 경우는 제외.
        result_filtered = result[np.where(result[:, 4] > score_threshold)]

        # 해당 클래스 별로 Detect된 여러개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수만큼 iteration해서 개별 오브젝트의 좌표값 추출.
        for i in range(len(result_filtered)):
            # 좌상단, 우하단 좌표 추출.
            left = int(result_filtered[i, 0])
            top = int(result_filtered[i, 1])
            right = int(result_filtered[i, 2])
            bottom = int(result_filtered[i, 3])
            caption = "{}: {:.4f}".format(labels_to_names_seq[result_ind], result_filtered[i, 4])
            cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.37, text_color, 1)
            if is_print:
                print(caption)

    return draw_img




if __name__ == "__main__":

    # Specify the path to model config and checkpoint file
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    #img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
    img = 'kitti_tiny/training/image_2/000068.jpeg'

    # 1.show_result_pyplot(model, img, results) 사용
    # results = inference_detector(model, img)
    # # visualize the results in a new window
    # show_result_pyplot(model, img, results)
    # #show_result_pyplot(model, img, results, score_thr = 0.3, title = 'park')


    # 2.만든 모듈 사용
    img_arr = cv2.imread(img)
    detected_img = get_detected_img(model, img_arr,  score_threshold=0.5, is_print=True)
    # detect 입력된 이미지는 bgr임. 이를 최종 출력시 rgb로 변환
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 12))
    plt.imshow(detected_img)
    plt.show()

