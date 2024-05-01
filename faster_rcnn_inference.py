from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
if __name__ == "__main__":
    # Config 설정하고 Pretrained 모델 다운로드
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'weights/epoch_12.pth'

    cfg = Config.fromfile(config_file)
    
    # dataset에 대한 환경 파라미터 수정.
    cfg.data.test.type = 'KittyTinyDataset'
    cfg.data.test.data_root = 'kitti_tiny'
    cfg.data.test.ann_file = 'val.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    # class의 갯수 수정.
    cfg.model.roi_head.bbox_head.num_classes = 4
    
    # pretrained 모델 로드 경로 설정
    cfg.load_from = checkpoint_file

    # ConfigDict' object has no attribute 'device 오류 대응
    cfg.device = 'cuda'

    # 모델 초기화 및 디바이스 설정
    model = init_detector(cfg, checkpoint_file, device=cfg.device)

    # 추론을 위한 이미지 경로 설정
    img = 'kitti_tiny/training/image_2/000068.jpeg'

    # 추론 수행
    result = inference_detector(model, img)

    # 결과 시각화
    show_result_pyplot(model, img, result, score_thr=0.3)
