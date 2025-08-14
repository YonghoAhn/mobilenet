import cv2
import websockets
import asyncio
import base64
import json
import time
import numpy as np
from collections import deque
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType)

# --- 설정 ---
SERVER_URI = "ws://192.168.0.2:3000"  # 접속할 서버 주소
VIDEO_PATH = "demo.mp4"             # 재생할 영상 파일 경로
TARGET_FPS = 30                      # 초당 전송할 프레임 수
FALL_MODEL_PATH = './mobilenet.hef'  # 낙상 감지 모델 경로
POSE_MODEL_PATH = './yolov8m_pose.hef'  # 자세 추정 모델 경로
SLIDING_WINDOW_SIZE = 40            # 슬라이딩 윈도우 크기
SLIDING_WINDOW_STRIDE = 20          # 슬라이딩 스텝 크기
FALL_THRESHOLD = 0.7                # 낙상 판정 임계값

class HailoIntegratedProcessor:
    """Hailo 통합 처리기 (완전한 모델 분리)"""
    
    def __init__(self, fall_model_path, pose_model_path, 
                 window_size=40, window_stride=10, fall_threshold=0.5):
        
        self.fall_model_path = fall_model_path
        self.pose_model_path = pose_model_path
        
        # 키포인트 이름 매핑
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 슬라이딩 윈도우 설정
        self.window_size = window_size
        self.window_stride = window_stride
        self.fall_threshold = fall_threshold
        
        # 확률 버퍼
        self.all_probabilities = []
        self.frame_counter = 0
        self.last_window_update_frame = 0
        self.current_window = []
        
        print(f"✅ Hailo 통합 처리기 초기화 완료")
        print(f"   - 슬라이딩 윈도우 크기: {window_size}")
        print(f"   - 슬라이딩 스텝(stride): {window_stride}")
        print(f"   - 낙상 판정 임계값: {fall_threshold}")

    # 기존 HailoIntegratedProcessor 클래스 안에 아래 함수들을 추가하거나 교체하세요.

    def predict_pose(self, frame, confidence_threshold=0.5, iou_threshold=0.7):
        """
        자세 추정 예측 (C++ 로직 기반 완전 재작성)
        """
        setup_start = time.time()
        
        # 1. VDevice 생성 및 모델 로드 (이 부분은 성능을 위해 __init__으로 옮기는 것이 좋습니다)
        pose_target = VDevice()
        pose_hef = HEF(self.pose_model_path)
        pose_configure_params = ConfigureParams.create_from_hef(
            hef=pose_hef, interface=HailoStreamInterface.PCIe
        )
        pose_network_groups = pose_target.configure(pose_hef, pose_configure_params)
        pose_network_group = pose_network_groups[0]
        pose_network_group_params = pose_network_group.create_params()
        
        pose_input_vstreams_params = InputVStreamParams.make(
            pose_network_group, format_type=FormatType.UINT8
        )
        # C++ 코드에서 uint16을 사용하므로, 여기에 맞춰야 할 수 있습니다.
        # 하지만 HEF에 따라 다르므로, 일단은 모델 정보를 따릅니다.
        pose_output_vstreams_params = OutputVStreamParams.make(
            pose_network_group, format_type=FormatType.UINT16
        )
        
        setup_time = time.time() - setup_start
        
        try:
            # 2. 전처리
            preprocess_start = time.time()
            # 모델의 입력 크기는 640x640 입니다.
            input_height, input_width = 640, 640
            processed_frame, ratio, (p_left, p_top) = self._preprocess_frame_for_pose(frame, (input_width, input_height))
            preprocess_time = time.time() - preprocess_start
            
            # 3. 추론 실행
            infer_start = time.time()
            with InferVStreams(pose_network_group,
                            pose_input_vstreams_params,
                            pose_output_vstreams_params) as pose_pipeline:
                with pose_network_group.activate(pose_network_group_params):
                    input_data = np.expand_dims(processed_frame, axis=0)
                    
                    # 입력 이름을 HEF 파일에서 직접 확인하는 것이 가장 좋습니다.
                    input_name = pose_hef.get_input_vstream_infos()[0].name
                    
                    infer_results_dict = pose_pipeline.infer({input_name: input_data})
                    
                    # C++ 로직처럼 이름으로 텐서를 정렬해야 할 수 있습니다.
                    # 출력 텐서 이름 순서가 중요합니다.
                    output_names = [info.name for info in pose_hef.get_output_vstream_infos()]
                    sorted_names = sorted(output_names) # 이름을 정렬하여 순서를 보장합니다.
                    
                    # 정렬된 이름 순서대로 텐서와 양자화 정보를 가져옵니다.
                    tensors = [infer_results_dict[name] for name in sorted_names]
                    quant_infos = {info.name: info.quant_info for info in pose_hef.get_output_vstream_infos()}
                    sorted_quant_infos = [quant_infos[name] for name in sorted_names]
                    
            infer_time = time.time() - infer_start
            
            # 4. 후처리 (C++ 로직 번역)
            postprocess_start = time.time()
            
            strides = [8, 16, 32]
            regression_length = 15
            
            # C++의 get_boxes_scores_keypoints 와 동일한 로직
            raw_boxes, scores, raw_keypoints = self._separate_tensors(tensors, sorted_quant_infos)
            
            # C++의 decode_boxes_and_keypoints 와 동일한 로직
            decoded_detections = self._decode_all(raw_boxes, scores, raw_keypoints, strides, 
                                                (input_width, input_height), regression_length, confidence_threshold)
            
            # C++의 nms 와 동일한 로직
            final_detections = self._nms(decoded_detections, iou_threshold)
            
            # 최종 결과 포맷팅
            keypoints_json = []
            if final_detections:
                # 가장 신뢰도 높은 사람의 키포인트만 사용
                best_detection = final_detections[0] 
                # 원본 프레임 좌표로 복원
                kpts = best_detection['keypoints']
                kpts[:, 0] = (kpts[:, 0] - p_left) / ratio
                kpts[:, 1] = (kpts[:, 1] - p_top) / ratio

                for i in range(len(kpts)):
                    if kpts[i, 2] > confidence_threshold:
                        keypoints_json.append({
                            "part": self.keypoint_names[i],
                            "x": float(kpts[i, 0]),
                            "y": float(kpts[i, 1]),
                            "score": float(kpts[i, 2])
                        })

            postprocess_time = time.time() - postprocess_start
            
            total_time = setup_time + preprocess_time + infer_time + postprocess_time
            
            result = {"keypoints": keypoints_json}

            timing = {
                "setup_time": setup_time,
                "preprocess_time": preprocess_time,
                "infer_time": infer_time,
                "postprocess_time": postprocess_time,
                "total_time": total_time
            }

            final_results = {
                "keypoints": keypoints_json,
                "detections": final_detections # NMS를 거친 최종 감지 정보
            }
            
            # 'detections'가 포함된 final_results를 반환
            return final_results, timing, ratio, (p_left, p_top)   
            #return result, timing
                    
        except Exception as e:
            print(f"자세 추정 오류: {e}")
            import traceback
            traceback.print_exc()
            return {"keypoints": []}, {}
        
        finally:
            del pose_target, pose_hef

    # --- 아래 헬퍼 함수들을 HailoIntegratedProcessor 클래스 내부에 추가 ---

    def _preprocess_frame_for_pose(self, frame, target_size):
        """
        Letterbox 전처리. 원본 비율을 유지하면서 패딩을 추가합니다.
        C++의 letterbox 처리와 동일한 결과를 내도록 구현.
        """
        target_w, target_h = target_size
        h, w, _ = frame.shape
        
        ratio = min(target_w / w, target_h / h)
        
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 패딩 계산
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        p_top, p_bottom = delta_h // 2, delta_h - (delta_h // 2)
        p_left, p_right = delta_w // 2, delta_w - (delta_w // 2)
        
        # 패딩 적용 (회색)
        padded_frame = cv2.copyMakeBorder(resized_frame, p_top, p_bottom, p_left, p_right,
                                        cv2.BORDER_CONSTANT, value=[114, 114, 114])
                                        
        # BGR to RGB 및 양자화
        frame_rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        quantized_frame = frame_rgb.astype(np.uint8) # 모델이 UINT8 입력을 받는 경우
        
        return quantized_frame, ratio, (p_left, p_top)

    def _dequantize(self, tensor, quant_info):
        return (tensor.astype(np.float32) - quant_info.qp_zp) * quant_info.qp_scale

    def _separate_tensors(self, tensors, quant_infos):
        """C++의 get_boxes_scores_keypoints 역할"""
        num_levels = len(tensors) // 3
        raw_boxes = []
        scores = []
        raw_keypoints = []
        
        for i in range(num_levels):
            # 텐서 순서는 이름 정렬에 따라 box, keypoints, score 순일 수 있음
            # HEF 출력 순서를 확인하고 인덱스를 맞춰야 함
            # 여기서는 box, score, keypoints 순서로 가정 (0, 1, 2)
            raw_boxes.append({'tensor': tensors[i*3 + 0], 'quant_info': quant_infos[i*3 + 0]})
            
            # Scores는 바로 dequantize
            score_tensor = self._dequantize(tensors[i*3 + 1], quant_infos[i*3 + 1])
            # (1, H, W, 1) -> (H*W, 1)
            scores.append(score_tensor.reshape(-1, 1))

            raw_keypoints.append({'tensor': tensors[i*3 + 2], 'quant_info': quant_infos[i*3 + 2]})
            
        return raw_boxes, np.concatenate(scores, axis=0), raw_keypoints

    def _decode_all(self, raw_boxes, scores, raw_keypoints, strides, net_dims, reg_len, conf_thresh):
        """C++의 decode_boxes_and_keypoints 역할"""
        net_w, net_h = net_dims
        detections = []
        
        # 각 그리드 셀의 중심 좌표 생성
        anchor_points = []
        strided_grids = []
        for stride in strides:
            grid_h, grid_w = net_h // stride, net_w // stride
            grid_y, grid_x = np.mgrid[0:grid_h, 0:grid_w]
            
            # C++의 (ct_col, ct_row) 와 동일
            center_x = (grid_x + 0.5) * stride
            center_y = (grid_y + 0.5) * stride
            
            # (H*W, 2)
            anchor = np.stack((center_x.flatten(), center_y.flatten()), axis=1)
            anchor_points.append(anchor)
            strided_grids.append((grid_h, grid_w))

        anchor_points = np.concatenate(anchor_points, axis=0)
        
        # 신뢰도 필터링
        candidate_indices = np.where(scores.flatten() > conf_thresh)[0]
        if len(candidate_indices) == 0:
            return []

        # DFL을 위한 거리 기댓값 계산 준비
        regression_distance = np.arange(reg_len + 1, dtype=np.float32)
        
        num_proposals = 0
        
        for i in range(len(strides)): # 각 stride 레벨에 대해
            
            # Box 디코딩
            box_tensor_info = raw_boxes[i]
            box_tensor = self._dequantize(box_tensor_info['tensor'], box_tensor_info['quant_info'])
            box_tensor = box_tensor.reshape(-1, 4, reg_len + 1) # (H*W, 4, 16)
            
            # Softmax
            box_dist_probs = self._softmax(box_tensor, axis=2)
            
            # 거리 기댓값 계산
            box_distances = np.sum(box_dist_probs * regression_distance, axis=2) * strides[i] # (H*W, 4)
            
            # Keypoint 디코딩
            kpt_tensor_info = raw_keypoints[i]
            kpt_tensor = self._dequantize(kpt_tensor_info['tensor'], kpt_tensor_info['quant_info'])
            kpt_tensor = kpt_tensor.reshape(-1, 17, 3) # (H*W, 17, 3)
            
            h, w = strided_grids[i]
            num_proposals_level = h * w
            
            for j in range(num_proposals_level):
                global_idx = num_proposals + j
                if scores[global_idx] < conf_thresh:
                    continue
                    
                anchor = anchor_points[global_idx]
                
                # Box 좌표 계산
                x1 = anchor[0] - box_distances[j, 0]
                y1 = anchor[1] - box_distances[j, 1]
                x2 = anchor[0] + box_distances[j, 2]
                y2 = anchor[1] + box_distances[j, 3]
                
                # Keypoint 좌표 계산
                kpt_raw = kpt_tensor[j] # (17, 3)
                kpt_xy_raw = kpt_raw[:, :2]
                kpt_vis_raw = kpt_raw[:, 2]
                
                # C++ 공식: strides[i] * (kpts_corrdinates - 0.5) + center_values
                # (kpts_corrdinates는 *2가 된 상태)
                kpt_x = (kpt_xy_raw[:, 0] * 2 - 0.5) * strides[i] + anchor[0]
                kpt_y = (kpt_xy_raw[:, 1] * 2 - 0.5) * strides[i] + anchor[1]
                kpt_vis = self._sigmoid(kpt_vis_raw)
                
                keypoints = np.stack((kpt_x, kpt_y, kpt_vis), axis=1)

                detections.append({
                    'box': [x1, y1, x2, y2],
                    'score': scores[global_idx, 0],
                    'keypoints': keypoints
                })
                
            num_proposals += num_proposals_level
            
        return detections

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _nms(self, detections, iou_threshold):
        if not detections:
            return []
            
        # 점수 기준으로 정렬
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        final_detections = []
        
        while detections:
            best_det = detections.pop(0)
            final_detections.append(best_det)
            
            # 남은 detection들과 IOU 계산
            remaining_detections = []
            for det in detections:
                iou = self._calculate_iou(best_det['box'], det['box'])
                if iou < iou_threshold:
                    remaining_detections.append(det)
            
            detections = remaining_detections
            
        return final_detections

    def _calculate_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def predict_fall(self, frame):
        """낙상 감지 예측 (완전 분리 방식)"""
        setup_start = time.time()
        
        # 1. VDevice 생성 및 모델 로드
        fall_target = VDevice()
        fall_hef = HEF(self.fall_model_path)
        fall_configure_params = ConfigureParams.create_from_hef(
            hef=fall_hef, interface=HailoStreamInterface.PCIe
        )
        fall_network_groups = fall_target.configure(fall_hef, fall_configure_params)
        fall_network_group = fall_network_groups[0]
        fall_network_group_params = fall_network_group.create_params()
        
        fall_input_vstreams_params = InputVStreamParams.make(
            fall_network_group, format_type=FormatType.UINT8
        )
        fall_output_vstreams_params = OutputVStreamParams.make(
            fall_network_group, format_type=FormatType.UINT8
        )
        
        fall_input_vstream_info = fall_hef.get_input_vstream_infos()[0]
        fall_output_vstream_info = fall_hef.get_output_vstream_infos()[0]
        
        setup_time = time.time() - setup_start
        
        try:
            # 2. 전처리
            preprocess_start = time.time()
            processed_frame = self._preprocess_frame_for_fall(frame)
            preprocess_time = time.time() - preprocess_start
            
            # 3. 추론 실행
            infer_start = time.time()
            with InferVStreams(fall_network_group,
                              fall_input_vstreams_params,
                              fall_output_vstreams_params) as fall_pipeline:
                with fall_network_group.activate(fall_network_group_params):
                    
                    # 배치 차원 추가 (1, 224, 224, 3)
                    input_data = np.expand_dims(processed_frame, axis=0)
                    input_dict = {fall_input_vstream_info.name: input_data}
                    
                    infer_results = fall_pipeline.infer(input_dict)
                    
                    # 출력 처리
                    output = infer_results[fall_output_vstream_info.name]
                    
                    # UINT8 출력을 FLOAT32로 역양자화
                    if output.dtype == np.uint8:
                        output_scale = 1.0 / 64.0
                        output_zero_point = 128
                        output_float = (output.astype(np.float32) - output_zero_point) * output_scale
                    else:
                        output_float = output.astype(np.float32)
                    
                    # Softmax 적용하여 확률 계산
                    if len(output_float.shape) > 1:
                        output_flat = output_float.flatten()
                    else:
                        output_flat = output_float
                    
                    if len(output_flat) >= 2:
                        logits = output_flat[:2]
                        exp_output = np.exp(logits - np.max(logits))
                        probs = exp_output / np.sum(exp_output)
                        emergency_prob = float(probs[1])  # Emergency 클래스 확률
                    else:
                        # 단일 출력인 경우 시그모이드 적용
                        emergency_prob = float(1 / (1 + np.exp(-output_flat[0])))
            
            infer_time = time.time() - infer_start
            total_time = setup_time + preprocess_time + infer_time
            
            timing = {
                "setup_time": setup_time,
                "preprocess_time": preprocess_time,
                "infer_time": infer_time,
                "total_time": total_time
            }
            
            return emergency_prob, timing
                    
        except Exception as e:
            print(f"낙상 감지 예측 오류: {e}")
            return 0.0, {"setup_time": 0, "preprocess_time": 0, "infer_time": 0, "total_time": 0}
        
        finally:
            # 4. 리소스 완전 해제 (중요!)
            del fall_target
            del fall_hef
    
    def process_frame(self, frame):
        """프레임 처리 (완전 분리 방식: 자세 추정 -> 낙상 감지)"""
        
        # 1. 자세 추정
        print(f"  🎯 자세 추정 시작...")
        pose_data, pose_timing, ratio, (p_left, p_top) = self.predict_pose(frame)
        print(f"  ✅ 자세 추정 완료: {pose_timing['total_time']:.3f}s (설정:{pose_timing['setup_time']:.3f}s, 추론:{pose_timing['infer_time']:.3f}s)")
        
        # 2. 낙상 감지
        print(f"  🎯 낙상 감지 시작...")
        frame_fall_prob, fall_timing = self.predict_fall(frame)
        print(f"  ✅ 낙상 감지 완료: {fall_timing['total_time']:.3f}s (설정:{fall_timing['setup_time']:.3f}s, 추론:{fall_timing['infer_time']:.3f}s)")
        
        # 3. 확률 버퍼에 추가
        self.all_probabilities.append(frame_fall_prob)
        self.frame_counter += 1
        
        # 버퍼 크기 제한
        max_buffer_size = self.window_size + self.window_stride * 10
        if len(self.all_probabilities) > max_buffer_size:
            self.all_probabilities = self.all_probabilities[-max_buffer_size:]
        
        # 4. 윈도우 업데이트 (stride 적용)
        should_update_window = False
        if len(self.all_probabilities) >= self.window_size:
            if len(self.current_window) == 0 or \
               (self.frame_counter - self.last_window_update_frame) >= self.window_stride:
                should_update_window = True
                self.last_window_update_frame = self.frame_counter
        
        if should_update_window:
            self.current_window = self.all_probabilities[-self.window_size:]
        
        # 5. 윈도우 평균 계산
        if len(self.current_window) > 0:
            avg_fall_prob = np.mean(self.current_window)
            is_fall_detected = avg_fall_prob > self.fall_threshold
        else:
            avg_fall_prob = 0.0
            is_fall_detected = False
        
        # 6. 낙상 감지 데이터 구성
        fall_detection_data = {
            "current_frame_prob": float(frame_fall_prob),
            "window_avg_prob": float(avg_fall_prob),
            "window_size": len(self.current_window),
            "actual_window_size": self.window_size,
            "window_stride": self.window_stride,
            "frame_number": self.frame_counter,
            "window_updated": should_update_window,
            "is_fall_detected": bool(is_fall_detected),
            "threshold": self.fall_threshold,
            "window_probs": list(self.current_window)
        }
        
        # 7. 처리 시간 정보 통합
        timing_info = {
            "pose_inference_time": pose_timing['total_time'],
            "fall_inference_time": fall_timing['total_time'],
            "pose_setup_time": pose_timing['setup_time'],
            "fall_setup_time": fall_timing['setup_time'],
            "pose_infer_only": pose_timing['infer_time'],
            "fall_infer_only": fall_timing['infer_time'],
            "total_processing_time": pose_timing['total_time'] + fall_timing['total_time']
        }
        
        return pose_data, fall_detection_data, timing_info, ratio, (p_left, p_top)
    
    def _preprocess_frame_for_fall(self, frame):
        """낙상 감지용 프레임 전처리"""
        # 224x224로 리사이즈
        resized_frame = cv2.resize(frame, (224, 224))
        
        # BGR to RGB 변환
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # 정규화 적용
        frame_array = np.array(frame_rgb, dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        
        normalized_frame = (frame_array - mean) / std
        
        # 양자화를 위한 스케일링
        quantized_frame = ((normalized_frame + 2.5) / 5.0 * 255.0)
        quantized_frame = np.clip(quantized_frame, 0, 255).astype(np.uint8)
        
        return quantized_frame
    
    # def _preprocess_frame_for_pose(self, frame):
    #     """자세 추정용 프레임 전처리"""
    #     # YOLOv8 포즈 모델에 맞는 전처리 (보통 640x640)
    #     target_size = 640
    #     resized_frame = cv2.resize(frame, (target_size, target_size))
        
    #     # BGR to RGB 변환
    #     frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
    #     # 정규화 (0-255 범위를 0-1로)
    #     normalized_frame = frame_rgb.astype(np.float32) / 255.0
        
    #     # 다시 UINT8로 변환 (Hailo 양자화를 위해)
    #     quantized_frame = (normalized_frame * 255).astype(np.uint8)
        
    #     return quantized_frame
    
    def _parse_yolo_pose_output(self, output, confidence_threshold):
        """YOLOv8 pose 출력 파싱 (다양한 출력 형태 지원)"""
        keypoints_json = []
        
        try:
            print(f"    🔍 Pose 출력 형태: {output.shape}")
            print(f"    📊 출력 범위: [{output.min():.4f}, {output.max():.4f}]")
            
            # 4D 출력 처리: (1, H, W, C) - Feature Map 형태
            if len(output.shape) == 4:
                batch, height, width, channels = output.shape
                print(f"      4D Feature Map: B={batch}, H={height}, W={width}, C={channels}")
                
                # Feature map에서 키포인트 추출 방법 1: 최대값 위치 찾기
                if channels >= 17:  # 17개 키포인트 채널이 있는 경우
                    print(f"      키포인트 채널 감지 (첫 17개 채널 사용)")
                    
                    for kp_idx in range(min(17, channels)):
                        if kp_idx < len(self.keypoint_names):
                            # 각 키포인트 채널에서 최대값 위치 찾기
                            kp_heatmap = output[0, :, :, kp_idx]  # (H, W)
                            max_val = np.max(kp_heatmap)
                            
                            if max_val > confidence_threshold:
                                # 최대값 위치 찾기
                                max_pos = np.unravel_index(np.argmax(kp_heatmap), kp_heatmap.shape)
                                y_idx, x_idx = max_pos
                                
                                # 좌표를 원본 이미지 크기로 스케일링 (640x640 기준)
                                x_scaled = float(x_idx * 640 / width)
                                y_scaled = float(y_idx * 640 / height)
                                
                                keypoints_json.append({
                                    "part": self.keypoint_names[kp_idx],
                                    "x": x_scaled,
                                    "y": y_scaled,
                                    "score": float(max_val)
                                })
                
                # Feature map에서 키포인트 추출 방법 2: 평면화 후 재구성
                elif channels == 64:  # 64채널인 경우 다른 접근
                    print(f"      64채널 Feature Map - 대안 파싱 시도")
                    
                    # 평면화
                    flattened = output.flatten()
                    print(f"      평면화 크기: {len(flattened)}")
                    
                    # 17*3=51개 값 찾기 (여러 위치에서 시도)
                    for start_idx in range(0, min(len(flattened)-51, 1000), 100):
                        test_data = flattened[start_idx:start_idx+51]
                        
                        # 값의 범위가 합리적인지 확인
                        if np.max(test_data) <= 1.0 and np.min(test_data) >= 0.0:
                            keypoints_data = test_data.reshape(17, 3)
                            temp_keypoints = []
                            
                            for i, (x, y, v) in enumerate(keypoints_data):
                                if i < len(self.keypoint_names) and v > confidence_threshold:
                                    temp_keypoints.append({
                                        "part": self.keypoint_names[i],
                                        "x": float(x * 640),  # 정규화된 값이라 가정하고 스케일링
                                        "y": float(y * 640),
                                        "score": float(v)
                                    })
                            
                            if len(temp_keypoints) > len(keypoints_json):
                                keypoints_json = temp_keypoints
                                print(f"      위치 {start_idx}에서 {len(temp_keypoints)}개 키포인트 발견")
                                break
            
            # 3D 출력 처리: (batch, detections, data)
            elif len(output.shape) == 3:
                batch_size, num_detections, data_size = output.shape
                print(f"      3D 출력: B={batch_size}, 검출={num_detections}, 데이터={data_size}")
                
                if num_detections > 0:
                    # 첫 번째 검출 결과 사용
                    detection = output[0, 0, :]  # 첫 번째 배치, 첫 번째 검출
                    
                    # YOLO pose 형태: [x1, y1, x2, y2, conf, cls, kp1_x, kp1_y, kp1_v, ...]
                    if len(detection) >= 56:  # 6(bbox+conf+cls) + 17*3(keypoints) = 57
                        keypoints_start = 6  # bbox(4) + conf(1) + cls(1) = 6
                        keypoints_data = detection[keypoints_start:keypoints_start + 51]  # 17 * 3
                        keypoints_reshaped = keypoints_data.reshape(17, 3)
                        
                        for i, (x, y, v) in enumerate(keypoints_reshaped):
                            if i < len(self.keypoint_names) and v > confidence_threshold:
                                keypoints_json.append({
                                    "part": self.keypoint_names[i],
                                    "x": float(x),
                                    "y": float(y),
                                    "score": float(v)
                                })
            
            # 2D 출력 처리: (detections, data)
            elif len(output.shape) == 2:
                num_detections, data_size = output.shape
                print(f"      2D 출력: 검출={num_detections}, 데이터={data_size}")
                
                if num_detections > 0:
                    detection = output[0, :]  # 첫 번째 검출
                    
                    if len(detection) >= 56:
                        keypoints_start = 6
                        keypoints_data = detection[keypoints_start:keypoints_start + 51]
                        keypoints_reshaped = keypoints_data.reshape(17, 3)
                        
                        for i, (x, y, v) in enumerate(keypoints_reshaped):
                            if i < len(self.keypoint_names) and v > confidence_threshold:
                                keypoints_json.append({
                                    "part": self.keypoint_names[i],
                                    "x": float(x),
                                    "y": float(y),
                                    "score": float(v)
                                })
            
            # 1D 출력 처리: 평면화된 출력
            elif len(output.shape) == 1:
                print(f"      1D 출력: 크기={output.shape[0]}")
                
                # 17개 키포인트 * 3 = 51개 값 찾기
                if len(output) >= 51:
                    keypoints_data = output[:51].reshape(17, 3)
                    
                    for i, (x, y, v) in enumerate(keypoints_data):
                        if i < len(self.keypoint_names) and v > confidence_threshold:
                            keypoints_json.append({
                                "part": self.keypoint_names[i],
                                "x": float(x),
                                "y": float(y),
                                "score": float(v)
                            })
            
            else:
                print(f"      ⚠️ 지원되지 않는 출력 형태: {output.shape}")
                
                # 최후의 수단: 임의의 키포인트 생성 (테스트용)
                print(f"      🔧 테스트용 임의 키포인트 생성")
                for i in range(min(5, len(self.keypoint_names))):
                    keypoints_json.append({
                        "part": self.keypoint_names[i],
                        "x": float(320 + np.random.randint(-50, 50)),  # 화면 중앙 근처
                        "y": float(240 + np.random.randint(-50, 50)),
                        "score": 0.8  # 임의의 높은 신뢰도
                    })
        
        except Exception as e:
            print(f"    ❌ Pose 출력 파싱 오류: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"      ✅ 파싱된 키포인트: {len(keypoints_json)}개")
        
        # 키포인트가 하나도 없으면 디버깅 정보 출력
        if len(keypoints_json) == 0:
            print(f"      🔍 디버깅: 출력 샘플 값들")
            if len(output.shape) == 4:
                print(f"        첫 번째 채널 통계: min={output[0,:,:,0].min():.4f}, max={output[0,:,:,0].max():.4f}, mean={output[0,:,:,0].mean():.4f}")
                if output.shape[3] > 16:
                    print(f"        17번째 채널 통계: min={output[0,:,:,16].min():.4f}, max={output[0,:,:,16].max():.4f}, mean={output[0,:,:,16].mean():.4f}")
        
    def _parse_yolo_pose_keypoint_output(self, output, confidence_threshold):
        """51채널 키포인트 출력 전용 파싱 (스케일링 개선)"""
        keypoints_json = []
        
        try:
            print(f"      🎯 키포인트 출력 파싱: {output.shape}")
            print(f"      📊 값 범위: [{output.min():.4f}, {output.max():.4f}]")
            
            if len(output.shape) == 4:  # (1, H, W, 51)
                batch, height, width, channels = output.shape
                
                if channels == 51:  # 17 키포인트 × 3 (x, y, visibility)
                    print(f"        17개 키포인트 × 3 채널 감지")
                    
                    # 각 키포인트 처리 (3채널씩 묶어서)
                    for kp_idx in range(17):
                        if kp_idx < len(self.keypoint_names):
                            # 3채널: x, y, visibility
                            x_channel = kp_idx * 3 + 0
                            y_channel = kp_idx * 3 + 1
                            v_channel = kp_idx * 3 + 2
                            
                            # 각 채널에서 최대값 위치 및 값 찾기
                            v_heatmap = output[0, :, :, v_channel]  # visibility 히트맵
                            max_v = np.max(v_heatmap)
                            
                            # confidence threshold 적용 (정규화 고려)
                            threshold = confidence_threshold
                            if max_v > 1.0:  # 값이 정규화되지 않은 경우
                                threshold = confidence_threshold * np.max(output[0, :, :, v_channel:v_channel+17:3])
                            
                            if max_v > threshold:
                                # visibility가 높은 위치 찾기
                                max_pos = np.unravel_index(np.argmax(v_heatmap), v_heatmap.shape)
                                y_idx, x_idx = max_pos
                                
                                # 해당 위치에서 x, y 값 가져오기
                                x_val = output[0, y_idx, x_idx, x_channel]
                                y_val = output[0, y_idx, x_idx, y_channel]
                                
                                # 좌표 계산 - 개선된 방법
                                if output.max() <= 1.0:
                                    # 이미 정규화된 경우
                                    x_coord = x_val * 640.0
                                    y_coord = y_val * 640.0
                                    confidence = max_v
                                else:
                                    # 그리드 기반 좌표 계산
                                    x_coord = float(x_idx) * (640.0 / width)
                                    y_coord = float(y_idx) * (640.0 / height)
                                    
                                    # offset 적용 (값이 0-1 범위인 경우)
                                    if x_val <= width and y_val <= height:
                                        x_coord += (x_val / width) * (640.0 / width)
                                        y_coord += (y_val / height) * (640.0 / height)
                                    
                                    # confidence 정규화
                                    confidence = max_v / np.max(v_heatmap) if np.max(v_heatmap) > 0 else 0.0
                                
                                # 좌표 범위 검증 및 클리핑
                                x_coord = np.clip(x_coord, 0, 640)
                                y_coord = np.clip(y_coord, 0, 640)
                                confidence = np.clip(confidence, 0, 1)
                                
                                keypoints_json.append({
                                    "part": self.keypoint_names[kp_idx],
                                    "x": float(x_coord),
                                    "y": float(y_coord),
                                    "score": float(confidence)
                                })
                                
                                print(f"          {self.keypoint_names[kp_idx]}: ({x_coord:.1f}, {y_coord:.1f}) conf={confidence:.3f}")
                
                else:
                    print(f"        ⚠️ 예상되지 않은 채널 수: {channels} (51 예상)")
            
            elif len(output.shape) == 3:  # (H, W, 51) - 배치 차원 없음
                height, width, channels = output.shape
                print(f"        3D 키포인트 출력: H={height}, W={width}, C={channels}")
                
                if channels == 51:
                    for kp_idx in range(17):
                        if kp_idx < len(self.keypoint_names):
                            x_channel = kp_idx * 3 + 0
                            y_channel = kp_idx * 3 + 1
                            v_channel = kp_idx * 3 + 2
                            
                            v_heatmap = output[:, :, v_channel]
                            max_v = np.max(v_heatmap)
                            
                            threshold = confidence_threshold
                            if max_v > 1.0:
                                threshold = confidence_threshold * np.max(output[:, :, v_channel:v_channel+17:3])
                            
                            if max_v > threshold:
                                max_pos = np.unravel_index(np.argmax(v_heatmap), v_heatmap.shape)
                                y_idx, x_idx = max_pos
                                
                                x_val = output[y_idx, x_idx, x_channel]
                                y_val = output[y_idx, x_idx, y_channel]
                                
                                if output.max() <= 1.0:
                                    x_coord = x_val * 640.0
                                    y_coord = y_val * 640.0
                                    confidence = max_v
                                else:
                                    x_coord = float(x_idx) * (640.0 / width)
                                    y_coord = float(y_idx) * (640.0 / height)
                                    
                                    if x_val <= width and y_val <= height:
                                        x_coord += (x_val / width) * (640.0 / width)
                                        y_coord += (y_val / height) * (640.0 / height)
                                    
                                    confidence = max_v / np.max(v_heatmap) if np.max(v_heatmap) > 0 else 0.0
                                
                                x_coord = np.clip(x_coord, 0, 640)
                                y_coord = np.clip(y_coord, 0, 640)
                                confidence = np.clip(confidence, 0, 1)
                                
                                keypoints_json.append({
                                    "part": self.keypoint_names[kp_idx],
                                    "x": float(x_coord),
                                    "y": float(y_coord),
                                    "score": float(confidence)
                                })
            
            else:
                print(f"        ⚠️ 지원되지 않는 키포인트 출력 형태: {output.shape}")
        
        except Exception as e:
            print(f"        ❌ 키포인트 파싱 오류: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"        ✅ 추출된 키포인트: {len(keypoints_json)}개")
        return keypoints_json
    
    def cleanup(self):
        """리소스 정리 (이미 del로 처리되므로 추가 작업 없음)"""
        print("✅ Hailo 통합 처리기 정리 완료 (모델별 개별 해제됨)")

async def stream_video_with_hailo_integration():
    """Hailo 통합 모델을 사용한 영상 스트리밍 (완전 분리 방식)"""
    print(f"서버에 연결 시도 중: {SERVER_URI}")
    
    # Hailo 통합 처리기 초기화
    processor = HailoIntegratedProcessor(
        fall_model_path=FALL_MODEL_PATH,
        pose_model_path=POSE_MODEL_PATH,
        window_size=SLIDING_WINDOW_SIZE,
        window_stride=SLIDING_WINDOW_STRIDE,
        fall_threshold=FALL_THRESHOLD
    )
    
    try:
        async with websockets.connect(SERVER_URI) as websocket:
            print("✅ 서버에 성공적으로 연결되었습니다.")
            print("=" * 80)
            
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                print(f"오류: '{VIDEO_PATH}' 영상을 열 수 없습니다.")
                return

            frame_count = 0
            fall_alert_sent = False
            
            while cap.isOpened():
                # 1. 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    print("\n영상을 모두 재생했습니다. 처음부터 다시 시작합니다.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    processor.all_probabilities.clear()
                    processor.current_window.clear()
                    processor.frame_counter = 0
                    processor.last_window_update_frame = 0
                    fall_alert_sent = False
                    continue

                # 2. 프레임 리사이즈
                resized_frame = cv2.resize(frame, (640, 480))
                
                # 3. 프레임 인코딩 (Base64)
                # _, buffer = cv2.imencode('.jpg', resized_frame)
                # frame_b64 = base64.b64encode(buffer).decode('utf-8')
                # frame_data_uri = f"data:image/jpeg;base64,{frame_b64}"

                print(f"\n🔄 Frame {frame_count:04d} 처리 시작")
                
                # 4. Hailo 통합 처리 (완전 분리 방식)
                frame_start_time = time.time()
                pose_data, fall_detection_data, timing_info, ratio, (p_left, p_top) = processor.process_frame(resized_frame)
                frame_total_time = time.time() - frame_start_time

                if 'detections' in pose_data:
                    for det in pose_data['detections']:
                        box = det['box']  # [x1, y1, x2, y2]

                        # 640x640 모델 입력 좌표를 -> 640x480 영상 좌표로 변환
                        x1 = int((box[0] - p_left) / ratio)
                        y1 = int((box[1] - p_top) / ratio)
                        x2 = int((box[2] - p_left) / ratio)
                        y2 = int((box[3] - p_top) / ratio)

                        # BGR 형식의 파란색 (255, 0, 0), 굵기 3
                        color = (255, 0, 0) 
                        thickness = 3

                        # resized_frame에 사각형 그리기
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, thickness)
                _, buffer = cv2.imencode('.jpg', resized_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frame_data_uri = f"data:image/jpeg;base64,{frame_b64}"

                # 자세 데이터 검증 및 디버깅
                print(f"      🔍 자세 데이터 검증:")
                print(f"        키포인트 개수: {len(pose_data['keypoints'])}")
                if len(pose_data['keypoints']) > 0:
                    sample_kp = pose_data['keypoints'][0]
                    print(f"        샘플 키포인트: {sample_kp}")
                    
                    # 좌표 범위 검증
                    x_coords = [kp['x'] for kp in pose_data['keypoints']]
                    y_coords = [kp['y'] for kp in pose_data['keypoints']]
                    print(f"        X 좌표 범위: [{min(x_coords):.1f}, {max(x_coords):.1f}]")
                    print(f"        Y 좌표 범위: [{min(y_coords):.1f}, {max(y_coords):.1f}]")
                    
                    # 신뢰도 범위 검증
                    scores = [kp['score'] for kp in pose_data['keypoints']]
                    print(f"        신뢰도 범위: [{min(scores):.3f}, {max(scores):.3f}]")
                
                # 원본 YOLO 형식과 동일하게 보장
                validated_pose_data = {
                    "keypoints": []
                }
                
                # 각 키포인트 검증 및 정규화
                for kp in pose_data['keypoints']:
                    # 필수 필드 검증
                    if all(key in kp for key in ['part', 'x', 'y', 'score']):
                        # 좌표 범위 검증 (0-640 범위)
                        x = float(kp['x'])
                        y = float(kp['y'])
                        score = float(kp['score'])
                        
                        # 좌표가 유효한 범위에 있는지 확인
                        if 0 <= x <= 640 and 0 <= y <= 640 and 0 <= score <= 1:
                            validated_pose_data["keypoints"].append({
                                "part": str(kp['part']),
                                "x": x,
                                "y": y,
                                "score": score
                            })
                        else:
                            print(f"        ⚠️ 범위 초과 키포인트 제외: {kp['part']} ({x:.1f}, {y:.1f}) score={score:.3f}")
                    else:
                        print(f"        ⚠️ 필수 필드 누락 키포인트 제외: {kp}")

                print(f"        ✅ 검증된 키포인트: {len(validated_pose_data['keypoints'])}개")

                # 5. 서버로 보낼 JSON 페이로드 구성
                payload = {
                    "type": "integrated_stream",  # 기존 서버와 호환되도록 변경
                    "timestamp": time.time(),
                    "frame_number": frame_count,
                    "frame": frame_data_uri,
                    "pose": validated_pose_data,  # 검증된 자세 데이터 사용
                    "fall_detection": fall_detection_data,
                    "timing": timing_info,
                    "hailo_info": {  # Hailo 관련 정보를 별도 필드로
                        "model_type": "hailo_integrated",
                        "fall_model": "mobilenet.hef",
                        "pose_model": "yolov8m_pose.hef"
                    }
                }

                # 6. 페이로드 크기 확인 및 최적화
                payload_size = len(json.dumps(payload))
                if payload_size > 1024 * 1024:  # 1MB 초과시 경고
                    print(f"      ⚠️ 페이로드 크기가 큼: {payload_size/1024:.1f}KB")
                    
                    # window_probs 제거하여 크기 줄이기
                    if "window_probs" in fall_detection_data:
                        del fall_detection_data["window_probs"]
                        payload["fall_detection"] = fall_detection_data
                        print(f"      🔧 window_probs 제거하여 크기 최적화")

                # 7. 데이터 전송 (에러 처리 추가)
                try:
                    await websocket.send(json.dumps(payload))
                    print(f"      ✅ 데이터 전송 성공 ({payload_size/1024:.1f}KB)")
                except Exception as send_error:
                    print(f"      ❌ 데이터 전송 실패: {send_error}")
                    
                    # 간소화된 페이로드로 재시도
                    simple_payload = {
                        "type": "integrated_stream",
                        "timestamp": time.time(),
                        "frame_number": frame_count,
                        "frame": frame_data_uri,
                        "pose": {
                            "keypoints": pose_data["keypoints"][:5]  # 처음 5개만
                        },
                        "fall_detection": {
                            "current_frame_prob": fall_detection_data["current_frame_prob"],
                            "window_avg_prob": fall_detection_data["window_avg_prob"],
                            "is_fall_detected": fall_detection_data["is_fall_detected"]
                        }
                    }
                    
                    try:
                        await websocket.send(json.dumps(simple_payload))
                        print(f"      ✅ 간소화 데이터 전송 성공")
                    except Exception as simple_send_error:
                        print(f"      ❌ 간소화 데이터 전송도 실패: {simple_send_error}")
                
                # 8. 이미지 저장 (디버깅용)
                if frame_count % 10 == 0:  # 10프레임마다 저장
                    debug_image_path = f"debug_frame_{frame_count:04d}.jpg"
                    cv2.imwrite(debug_image_path, resized_frame)
                    print(f"      💾 디버그 이미지 저장: {debug_image_path}")
                
                # 9. Base64 이미지 검증 (추가 디버깅)
                if frame_count == 1:  # 첫 프레임에서만 검증
                    print(f"      🔍 Base64 이미지 검증:")
                    print(f"        인코딩 전 이미지: {resized_frame.shape}, {resized_frame.dtype}")
                    print(f"        JPEG 버퍼 크기: {len(buffer)} bytes")
                    print(f"        Base64 문자열 길이: {len(frame_b64)}")
                    print(f"        Data URI 길이: {len(frame_data_uri)}")
                    print(f"        Data URI 시작: {frame_data_uri[:50]}...")
                    
                    # Base64 디코딩 테스트 (import 충돌 해결)
                    try:
                        decoded_img = base64.b64decode(frame_b64)
                        print(f"        ✅ Base64 디코딩 성공: {len(decoded_img)} bytes")
                    except Exception as decode_error:
                        print(f"        ❌ Base64 디코딩 실패: {decode_error}")
                
                # 8. 콘솔 출력 (상세)
                keypoints_count = len(validated_pose_data["keypoints"])
                high_confidence_count = len([kp for kp in validated_pose_data["keypoints"] if kp["score"] > 0.5])
                
                status_symbol = "🔴" if fall_detection_data["is_fall_detected"] else "🟢"
                window_update_symbol = "↻" if fall_detection_data["window_updated"] else " "
                
                print(f"📊 {time.strftime('%H:%M:%S')} Frame {frame_count:04d} {status_symbol}{window_update_symbol}")
                print(f"   Pose[Hailo]: {keypoints_count} pts (HC: {high_confidence_count})")
                print(f"   Fall[Hailo]: {fall_detection_data['current_frame_prob']:.3f} (Win[{fall_detection_data['window_size']}]: {fall_detection_data['window_avg_prob']:.3f})")
                print(f"   Timing: P={timing_info['pose_inference_time']:.3f}s F={timing_info['fall_inference_time']:.3f}s Total={timing_info['total_processing_time']:.3f}s")
                print(f"   Setup: P={timing_info['pose_setup_time']:.3f}s F={timing_info['fall_setup_time']:.3f}s")
                print(f"   실제 프레임 시간: {frame_total_time:.3f}s")
                
                # 키포인트 상세 정보 (처음 3개 프레임만)
                if frame_count <= 3 and keypoints_count > 0:
                    print(f"   🔍 키포인트 상세 (Frame {frame_count}):")
                    for i, kp in enumerate(validated_pose_data["keypoints"][:5]):  # 처음 5개만
                        print(f"     {kp['part']}: ({kp['x']:.1f}, {kp['y']:.1f}) conf={kp['score']:.3f}")
                
                # WebSocket 페이로드 크기 체크
                payload_str = json.dumps(payload)
                payload_size = len(payload_str)
                print(f"   📦 페이로드: {payload_size/1024:.1f}KB, 키포인트: {keypoints_count}개")
                
                # 8. 낙상 감지 알림
                if fall_detection_data["is_fall_detected"] and not fall_alert_sent:
                    print("\n⚠️  [Hailo] 낙상 감지됨! ⚠️")
                    fall_alert_sent = True
                elif not fall_detection_data["is_fall_detected"] and fall_alert_sent:
                    print("\n✅ [Hailo] 정상 상태로 복귀")
                    fall_alert_sent = False
                
                frame_count += 1

                # 9. FPS 제어
                await asyncio.sleep(1 / TARGET_FPS)

            cap.release()
    
    except Exception as e:
        print(f"스트리밍 중 오류: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 10. 리소스 정리
        processor.cleanup()

# --- 원본 YOLO vs Hailo 비교 테스트 ---
async def compare_pose_formats():
    """원본 YOLO와 Hailo 자세 데이터 형식 비교"""
    print("🔍 자세 데이터 형식 비교 테스트")
    
    # 원본 YOLO 형식 시뮬레이션
    original_yolo_format = {
        "keypoints": [
            {"part": "nose", "x": 320.5, "y": 180.2, "score": 0.857},
            {"part": "left_eye", "x": 310.1, "y": 175.8, "score": 0.782},
            {"part": "right_eye", "x": 330.7, "y": 176.3, "score": 0.798}
        ]
    }
    
    # Hailo 처리기로 테스트
    processor = HailoIntegratedProcessor(
        fall_model_path=FALL_MODEL_PATH,
        pose_model_path=POSE_MODEL_PATH
    )
    
    # 테스트 프레임 생성
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (128, 128, 128)  # 회색 배경
    
    try:
        pose_data, _, _, _, _ = processor.process_frame(test_frame)
        
        print(f"📊 형식 비교:")
        print(f"  원본 YOLO 키포인트: {len(original_yolo_format['keypoints'])}개")
        print(f"  Hailo 키포인트: {len(pose_data['keypoints'])}개")
        
        if len(pose_data['keypoints']) > 0:
            print(f"  원본 형식 예시: {original_yolo_format['keypoints'][0]}")
            print(f"  Hailo 형식 예시: {pose_data['keypoints'][0]}")
            
            # 필드 비교
            original_keys = set(original_yolo_format['keypoints'][0].keys())
            hailo_keys = set(pose_data['keypoints'][0].keys())
            
            print(f"  필드 일치: {original_keys == hailo_keys}")
            if original_keys != hailo_keys:
                print(f"    원본 필드: {original_keys}")
                print(f"    Hailo 필드: {hailo_keys}")
                print(f"    누락 필드: {original_keys - hailo_keys}")
                print(f"    추가 필드: {hailo_keys - original_keys}")
        
    except Exception as e:
        print(f"❌ 비교 테스트 실패: {e}")
    
    finally:
        processor.cleanup()
    
    print("✅ 자세 데이터 형식 비교 완료")

# --- 이미지 전용 테스트 함수 ---
async def test_image_only():
    """이미지 전송만 테스트 (Hailo 처리 없이)"""
    print("🧪 이미지 전용 테스트 시작")
    
    async with websockets.connect(SERVER_URI) as websocket:
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        for i in range(5):  # 5프레임만 테스트
            ret, frame = cap.read()
            if not ret:
                break
                
            resized_frame = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_data_uri = f"data:image/jpeg;base64,{frame_b64}"
            
            # 최소한의 페이로드
            test_payload = {
                "type": "cctv_stream",  # 원본 타입 사용
                "frame": frame_data_uri,
                "pose": {"keypoints": []}  # 빈 포즈 데이터
            }
            
            await websocket.send(json.dumps(test_payload))
            print(f"  📤 테스트 프레임 {i+1} 전송 완료")
            await asyncio.sleep(1)
        
        cap.release()
    print("✅ 이미지 전용 테스트 완료")
# --- 스크립트 실행 ---
if __name__ == "__main__":
    print("=== Hailo 완전 분리 통합 실시간 스트리밍 ===")
    print(f"낙상 감지 모델: {FALL_MODEL_PATH}")
    print(f"자세 추정 모델: {POSE_MODEL_PATH}")
    print(f"영상: {VIDEO_PATH}")
    print(f"FPS: {TARGET_FPS}")
    print(f"슬라이딩 윈도우: {SLIDING_WINDOW_SIZE} 프레임")
    print(f"슬라이딩 스텝: {SLIDING_WINDOW_STRIDE} 프레임")
    print(f"낙상 임계값: {FALL_THRESHOLD}")
    print("📝 참고: 각 프레임마다 모델이 완전히 로드/해제됩니다.")
    print("=" * 60)
    
    # 실행 모드 선택
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "test":
            print("🧪 이미지 전용 테스트 모드")
            try:
                asyncio.run(test_image_only())
            except Exception as e:
                print(f"테스트 실패: {e}")
        elif mode == "compare":
            print("🔍 자세 데이터 형식 비교 모드")
            try:
                asyncio.run(compare_pose_formats())
            except Exception as e:
                print(f"비교 테스트 실패: {e}")
        else:
            print(f"❌ 알 수 없는 모드: {mode}")
            print("사용법: python script.py [test|compare]")
    else:
        print("🚀 전체 Hailo 통합 모드")
        try:
            asyncio.run(stream_video_with_hailo_integration())
        except ConnectionRefusedError:
            print("오류: 서버 연결이 거부되었습니다. Node.js 서버가 실행 중인지 확인하세요.")
        except KeyboardInterrupt:
            print("\n스트리밍을 중단합니다.")
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다 - {e}")
            print("HEF 모델 파일과 비디오 파일 경로를 확인하세요.")
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            import traceback
            traceback.print_exc()(f"영상: {VIDEO_PATH}")
    print(f"FPS: {TARGET_FPS}")
    print(f"슬라이딩 윈도우: {SLIDING_WINDOW_SIZE} 프레임")
    print(f"슬라이딩 스텝: {SLIDING_WINDOW_STRIDE} 프레임")
    print(f"낙상 임계값: {FALL_THRESHOLD}")
    print("📝 참고: 각 프레임마다 모델이 완전히 로드/해제됩니다.")
    print("=" * 60)
    
    try:
        asyncio.run(stream_video_with_hailo_integration())
    except ConnectionRefusedError:
        print("오류: 서버 연결이 거부되었습니다. Node.js 서버가 실행 중인지 확인하세요.")
    except KeyboardInterrupt:
        print("\n스트리밍을 중단합니다.")
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e}")
        print("HEF 모델 파일과 비디오 파일 경로를 확인하세요.")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()