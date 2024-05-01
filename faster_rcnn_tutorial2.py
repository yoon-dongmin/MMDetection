from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2




if __name__ == "__main__":


    # Specify the path to model config and checkpoint file
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    video = 'demo/demo.mp4'
    video_reader = mmcv.VideoReader(video)


    # or img = mmcv.imread(img), which will only load it once
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('demo/demo_out.mp4', fourcc, video_reader.fps,(video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        frame = model.show_result(frame, result, score_thr=0.4)

        video_writer.write(frame)

    if video_writer:
            video_writer.release()