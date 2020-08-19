import logging

from action.action_Identify import ActionIdentify
from action.actions import *
from deep_sort import DeepSort
from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    model = Darknet("config/yolo-obj.cfg", img_size=(608, 608))
    model.load_darknet_weights("weights/yolo-obj_last.weights")
    model.to("cuda:0")

    # 跟踪器
    tracker = DeepSort("weights/ckpt.t7",
                       min_confidence=1,
                       use_cuda=True,
                       nn_budget=30,
                       n_init=3,
                       max_iou_distance=0.7,
                       max_dist=0.3,
                       max_age=30)

    # Action Identify
    # action_id = ActionIdentify(actions=[TakeOff(4, delta=(0, 1)),
    #                                     Landing(4, delta=(2, 2)),
    #                                     Glide(4, delta=(1, 2)),
    #                                     FastCrossing(4, speed=0.2),
    #                                     BreakInto(0, timeout=2)],
    #                            max_age=30,
    #                            max_size=8)

    video_detector = VideoDetector(model, "config/obj.names",
                                   #font_path="font/Noto_Serif_SC/NotoSerifSC-Regular.otf",
                                   #font_size=14,
                                   thickness=2,
                                   skip_frames=2,
                                   thres=0.3,
                                   class_mask=[0, 1, 2, 3, 4],
                                   nms_thres=0.4,
                                   tracker=tracker,
                                   half=True)

    for image, detections, _ in video_detector.detect("test_vdo/kitti_road_1.mp4",
                                                      output_path= "data_tracker/kitti_road_output_2.mp4",
                                                      real_show=False,
						      skip_secs=0):
        # print(detections)
        pass
