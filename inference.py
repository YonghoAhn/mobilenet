import os
import time
import numpy as np
import cv2
from PIL import Image
from glob import glob
import pandas as pd
from tqdm import tqdm
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType)

class HailoFallDetectionTester:
    """Hailo 낙상 감지 모델 테스터"""
    
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.target = None
        self.hef = None
        self.network_group = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        
        self._initialize_hef()
    
    def _initialize_hef(self):
        """HEF 모델 초기화"""
        print(f"🔄 Hailo 모델 초기화: {self.hef_path}")
        
        self.target = VDevice()
        self.hef = HEF(self.hef_path)
        
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = network_groups[0]
        
        # 입출력 정보 가져오기
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]

        print(f"  📋 모델 정보:")
        print(f"    입력: {self.input_vstream_info.name}")
        print(f"    입력 형태: {self.input_vstream_info.shape} ({self.input_vstream_info.format.type})")
        print(f"    출력: {self.output_vstream_info.name}")
        print(f"    출력 형태: {self.output_vstream_info.shape} ({self.output_vstream_info.format.type})")
        print("✅ Hailo 모델 초기화 완료")
    
    def __del__(self):
        """소멸자"""
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'target') and self.target is not None:
            try:
                del self.target
                self.target = None
                print("  🧹 Hailo VDevice 리소스 정리 완료")
            except Exception as e:
                print(f"  ⚠️ Hailo VDevice 정리 중 오류: {e}")
    
    def extract_frames_from_video(self, video_path, num_frames=224):
        """비디오에서 224개 프레임 추출"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        
        # 균등하게 프레임 선택
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB 변환 및 224x224 리사이즈
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        return frames
    
    def preprocess_frames_for_hailo(self, frames):
        """Hailo 입력용 프레임 전처리"""
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 224
        elif len(frames) < 224:
            # 프레임이 부족한 경우 반복
            while len(frames) < 224:
                frames.extend(frames[:min(len(frames), 224 - len(frames))])
            frames = frames[:224]
        elif len(frames) > 224:
            frames = frames[:224]
        
        # numpy 배열로 변환 및 정규화
        frames_array = np.array(frames, dtype=np.float32)  # (224, 224, 224, 3)
        
        # ImageNet 정규화 적용
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        
        # 정규화 적용
        frames_array = (frames_array - mean) / std
        
        # 다시 0-255 범위로 스케일링 (양자화를 위해)
        frames_array = ((frames_array + 2.5) / 5.0 * 255.0)  # 대략적인 스케일링
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
        
        return frames_array
    
    def predict_video_frames(self, video_path):
        """비디오의 모든 프레임에 대해 예측"""
        
        # 프레임 추출
        frames = self.extract_frames_from_video(video_path, num_frames=224)
        print(f"  📹 추출된 프레임 수: {len(frames)}")
        
        # 전처리
        processed_frames = self.preprocess_frames_for_hailo(frames)
        print(f"  🔧 전처리된 프레임: {processed_frames.shape}")
        print(f"  📊 메모리 크기: {processed_frames.nbytes:,} bytes")
        print(f"  📊 값 범위: [{processed_frames.min()}, {processed_frames.max()}]")
        
        # Hailo 추론
        network_group_params = self.network_group.create_params()
        
        input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        
        with InferVStreams(self.network_group,
                          input_vstreams_params,
                          output_vstreams_params) as pipeline:
            with self.network_group.activate(network_group_params):
                
                input_dict = {self.input_vstream_info.name: processed_frames}
                
                print(f"  🚀 Hailo 추론 실행 중...")
                infer_results = pipeline.infer(input_dict)
                
                # 출력 처리
                output = infer_results[self.output_vstream_info.name]
                print(f"  📤 Raw 출력: {output.shape}, dtype: {output.dtype}")
                print(f"  📊 출력 범위: [{output.min()}, {output.max()}]")
                
                # UINT8 출력을 FLOAT32로 역양자화
                if output.dtype == np.uint8:
                    output_scale = 1.0 / 64.0
                    output_zero_point = 128
                    output_float = (output.astype(np.float32) - output_zero_point) * output_scale
                else:
                    output_float = output.astype(np.float32)
                
                print(f"  🔄 역양자화 출력: {output_float.shape}")
                print(f"  📊 역양자화 범위: [{output_float.min():.4f}, {output_float.max():.4f}]")
                
                # 프레임별 결과 처리
                frame_results = []
                
                if len(output_float.shape) == 2 and output_float.shape[1] == 2:
                    # (224, 2) 형태 - 각 프레임별 2클래스 예측
                    print(f"  📊 프레임별 예측 처리: {output_float.shape[0]}개 프레임")
                    
                    for i in range(output_float.shape[0]):
                        frame_logits = output_float[i]  # (2,)
                        
                        # Softmax 적용
                        exp_output = np.exp(frame_logits - np.max(frame_logits))
                        probs = exp_output / np.sum(exp_output)
                        emergency_prob = float(probs[1])  # Emergency 클래스 확률
                        prediction = 1 if emergency_prob >= 0.5 else 0
                        
                        frame_results.append({
                            'frame_idx': i,
                            'emergency_prob': emergency_prob,
                            'prediction': prediction,
                            'logits': frame_logits.tolist()
                        })
                
                elif len(output_float.shape) == 1 and output_float.shape[0] == 2:
                    # (2,) 형태 - 단일 예측
                    exp_output = np.exp(output_float - np.max(output_float))
                    probs = exp_output / np.sum(exp_output)
                    emergency_prob = float(probs[1])
                    prediction = 1 if emergency_prob >= 0.5 else 0
                    
                    frame_results.append({
                        'frame_idx': 0,
                        'emergency_prob': emergency_prob,
                        'prediction': prediction,
                        'logits': output_float.tolist()
                    })
                
                else:
                    print(f"  ⚠️ 예상되지 않은 출력 형태: {output_float.shape}")
                    # 기본 처리
                    avg_logit = np.mean(output_float)
                    emergency_prob = float(1 / (1 + np.exp(-avg_logit)))
                    prediction = 1 if emergency_prob >= 0.5 else 0
                    
                    frame_results.append({
                        'frame_idx': 0,
                        'emergency_prob': emergency_prob,
                        'prediction': prediction,
                        'logits': output_float.flatten().tolist()
                    })
                
                return frame_results

def extract_label_from_filename(filename):
    """파일명에서 라벨 추출"""
    basename = os.path.basename(filename)
    if 'Emergency' in basename and 'Non_Emergency' not in basename:
        return 1
    elif 'Non_Emergency' in basename:
        return 0
    else:
        return 0

def test_hailo_model(hef_path, video_dir):
    """Hailo 모델 테스트"""
    
    print(f"🚀 Hailo 모델 테스트 시작")
    print(f"📁 HEF 경로: {hef_path}")
    print(f"📁 비디오 디렉토리: {video_dir}")
    print("=" * 80)
    
    # Hailo 모델 초기화
    tester = HailoFallDetectionTester(hef_path)
    
    # 비디오 파일 목록
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob(os.path.join(video_dir, '**', ext), recursive=True))
    video_files = sorted(video_files)
    
    print(f"📊 총 비디오 파일: {len(video_files)}개")
    
    results = []
    
    try:
        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                file_name = os.path.splitext(os.path.basename(video_file))[0]
                true_label = extract_label_from_filename(video_file)
                
                print(f"\n🔄 처리 중: {file_name}")
                
                # 프레임별 예측
                start_time = time.time()
                frame_results = tester.predict_video_frames(video_file)
                processing_time = time.time() - start_time
                
                # 전체 비디오에 대한 통계
                if frame_results:
                    emergency_probs = [r['emergency_prob'] for r in frame_results]
                    predictions = [r['prediction'] for r in frame_results]
                    
                    avg_prob = np.mean(emergency_probs)
                    max_prob = np.max(emergency_probs)
                    min_prob = np.min(emergency_probs)
                    emergency_ratio = np.mean(predictions)
                    
                    # 최종 비디오 예측 (평균 확률 기준)
                    video_prediction = 1 if avg_prob >= 0.5 else 0
                    is_correct = video_prediction == true_label
                    
                    print(f"  📊 프레임 분석:")
                    print(f"    처리된 프레임 수: {len(frame_results)}")
                    print(f"    평균 Emergency 확률: {avg_prob:.4f}")
                    print(f"    최대/최소 확률: {max_prob:.4f}/{min_prob:.4f}")
                    print(f"    Emergency 프레임 비율: {emergency_ratio:.2%}")
                    print(f"  🎯 비디오 예측: {video_prediction}, 실제: {true_label}, 정답: {is_correct}")
                    print(f"  ⏱️ 처리 시간: {processing_time:.2f}초")
                    
                    # 결과 저장
                    result = {
                        'file_name': file_name,
                        'true_label': true_label,
                        'video_prediction': video_prediction,
                        'avg_emergency_prob': avg_prob,
                        'max_emergency_prob': max_prob,
                        'min_emergency_prob': min_prob,
                        'emergency_frame_ratio': emergency_ratio,
                        'is_correct': is_correct,
                        'processing_time': processing_time,
                        'frame_results': frame_results
                    }
                    
                    results.append(result)
                else:
                    print("  ❌ 프레임 결과가 없음")
                
            except Exception as e:
                print(f"❌ {video_file} 처리 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    finally:
        # 리소스 정리
        tester.cleanup()
    
    # 결과 저장
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    if results:
        # 비디오 레벨 결과
        video_results = []
        for r in results:
            video_results.append({
                'file_name': r['file_name'],
                'true_label': r['true_label'],
                'video_prediction': r['video_prediction'],
                'avg_emergency_prob': r['avg_emergency_prob'],
                'max_emergency_prob': r['max_emergency_prob'],
                'min_emergency_prob': r['min_emergency_prob'],
                'emergency_frame_ratio': r['emergency_frame_ratio'],
                'is_correct': r['is_correct'],
                'processing_time': r['processing_time']
            })
        
        video_df = pd.DataFrame(video_results)
        video_csv = f"hailo_video_results_{timestamp}.csv"
        video_df.to_csv(video_csv, index=False)
        print(f"\n✅ 비디오 결과 저장: {video_csv}")
        
        # 프레임 레벨 결과
        frame_results = []
        for r in results:
            for frame_r in r['frame_results']:
                frame_entry = {
                    'file_name': r['file_name'],
                    'true_label': r['true_label'],
                    'frame_idx': frame_r['frame_idx'],
                    'emergency_prob': frame_r['emergency_prob'],
                    'prediction': frame_r['prediction']
                }
                
                # logits 정보 추가
                if len(frame_r['logits']) >= 2:
                    frame_entry['logits_0'] = frame_r['logits'][0]
                    frame_entry['logits_1'] = frame_r['logits'][1]
                elif len(frame_r['logits']) == 1:
                    frame_entry['logits_0'] = frame_r['logits'][0]
                    frame_entry['logits_1'] = 0.0
                else:
                    frame_entry['logits_0'] = 0.0
                    frame_entry['logits_1'] = 0.0
                
                frame_results.append(frame_entry)
        
        frame_df = pd.DataFrame(frame_results)
        frame_csv = f"hailo_frame_results_{timestamp}.csv"
        frame_df.to_csv(frame_csv, index=False)
        print(f"✅ 프레임 결과 저장: {frame_csv}")
        
        # 성능 요약
        accuracy = sum(1 for r in video_results if r['is_correct']) / len(video_results)
        avg_processing_time = np.mean([r['processing_time'] for r in video_results])
        
        print(f"\n📊 Hailo 모델 성능 요약:")
        print(f"  정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  평균 처리 시간: {avg_processing_time:.3f}초/비디오")
        print(f"  총 처리 비디오: {len(video_results)}개")
        
        # 프레임 레벨 통계
        if frame_results:
            total_frames = len(frame_results)
            emergency_frames = sum(1 for f in frame_results if f['prediction'] == 1)
            avg_frame_prob = np.mean([f['emergency_prob'] for f in frame_results])
            
            print(f"\n📊 프레임 레벨 통계:")
            print(f"  총 프레임 수: {total_frames:,}")
            print(f"  Emergency 예측 프레임: {emergency_frames:,} ({emergency_frames/total_frames:.2%})")
            print(f"  평균 Emergency 확률: {avg_frame_prob:.4f}")
    
    return results

def compare_models(pytorch_video_csv, pytorch_frame_csv, hailo_video_csv, hailo_frame_csv):
    """PyTorch와 Hailo 모델 결과 비교"""
    
    print("🔍 PyTorch vs Hailo 모델 비교 분석")
    print("=" * 60)
    
    try:
        # 비디오 레벨 비교
        pytorch_video = pd.read_csv(pytorch_video_csv)
        hailo_video = pd.read_csv(hailo_video_csv)
        
        # 공통 파일들만 비교
        common_files = set(pytorch_video['file_name']).intersection(set(hailo_video['file_name']))
        
        if common_files:
            print(f"📊 비교 가능한 비디오: {len(common_files)}개")
            
            pytorch_common = pytorch_video[pytorch_video['file_name'].isin(common_files)].sort_values('file_name')
            hailo_common = hailo_video[hailo_video['file_name'].isin(common_files)].sort_values('file_name')
            
            # 정확도 비교
            pytorch_acc = pytorch_common['is_correct'].mean()
            hailo_acc = hailo_common['is_correct'].mean()
            
            print(f"\n🎯 정확도 비교:")
            print(f"  PyTorch: {pytorch_acc:.4f} ({pytorch_acc*100:.2f}%)")
            print(f"  Hailo:   {hailo_acc:.4f} ({hailo_acc*100:.2f}%)")
            print(f"  차이:    {abs(pytorch_acc - hailo_acc):.4f}")
            
            # 확률 분포 비교
            pytorch_probs = pytorch_common['avg_emergency_prob']
            hailo_probs = hailo_common['avg_emergency_prob']
            
            prob_correlation = np.corrcoef(pytorch_probs, hailo_probs)[0, 1]
            prob_mae = np.mean(np.abs(pytorch_probs - hailo_probs))
            
            print(f"\n📊 확률 분포 비교:")
            print(f"  확률 상관관계: {prob_correlation:.4f}")
            print(f"  평균 절대 오차: {prob_mae:.4f}")
            print(f"  PyTorch 평균 확률: {pytorch_probs.mean():.4f}")
            print(f"  Hailo 평균 확률:   {hailo_probs.mean():.4f}")
            
            # 처리 시간 비교
            pytorch_time = pytorch_common['processing_time'].mean()
            hailo_time = hailo_common['processing_time'].mean()
            
            print(f"\n⏱️ 처리 시간 비교:")
            print(f"  PyTorch: {pytorch_time:.3f}초/비디오")
            print(f"  Hailo:   {hailo_time:.3f}초/비디오")
            print(f"  속도 향상: {pytorch_time/hailo_time:.2f}x" if hailo_time > 0 else "")
            
            # 불일치 사례 분석
            disagreements = pytorch_common['video_prediction'] != hailo_common['video_prediction']
            disagreement_count = disagreements.sum()
            
            if disagreement_count > 0:
                print(f"\n❗ 예측 불일치 사례: {disagreement_count}개")
                disagreement_files = pytorch_common[disagreements]['file_name'].tolist()
                for i, file_name in enumerate(disagreement_files[:5]):  # 상위 5개만 표시
                    pt_pred = pytorch_common[pytorch_common['file_name'] == file_name]['video_prediction'].iloc[0]
                    hailo_pred = hailo_common[hailo_common['file_name'] == file_name]['video_prediction'].iloc[0]
                    true_label = pytorch_common[pytorch_common['file_name'] == file_name]['true_label'].iloc[0]
                    print(f"    {i+1}. {file_name}: PyTorch={pt_pred}, Hailo={hailo_pred}, 실제={true_label}")
        
        # 프레임 레벨 비교 (선택적)
        if os.path.exists(pytorch_frame_csv) and os.path.exists(hailo_frame_csv):
            pytorch_frame = pd.read_csv(pytorch_frame_csv)
            hailo_frame = pd.read_csv(hailo_frame_csv)
            
            # 샘플 파일에 대한 프레임별 비교
            sample_file = list(common_files)[0] if common_files else None
            if sample_file:
                print(f"\n🔍 샘플 파일 프레임별 분석: {sample_file}")
                
                pt_frames = pytorch_frame[pytorch_frame['file_name'] == sample_file]
                hailo_frames = hailo_frame[hailo_frame['file_name'] == sample_file]
                
                if len(pt_frames) > 0 and len(hailo_frames) > 0:
                    pt_frame_acc = (pt_frames['prediction'] == pt_frames['true_label']).mean()
                    hailo_frame_acc = (hailo_frames['prediction'] == hailo_frames['true_label']).mean()
                    
                    print(f"  프레임별 정확도 - PyTorch: {pt_frame_acc:.4f}, Hailo: {hailo_frame_acc:.4f}")
                    
                    # 프레임별 확률 상관관계
                    min_frames = min(len(pt_frames), len(hailo_frames))
                    if min_frames > 1:
                        pt_probs = pt_frames['emergency_prob'].iloc[:min_frames]
                        hailo_probs = hailo_frames['emergency_prob'].iloc[:min_frames]
                        frame_corr = np.corrcoef(pt_probs, hailo_probs)[0, 1]
                        print(f"  프레임별 확률 상관관계: {frame_corr:.4f}")
        
    except Exception as e:
        print(f"❌ 모델 비교 중 오류: {e}")
        import traceback
        traceback.print_exc()

# 실행 예제
if __name__ == "__main__":
    # 설정
    hef_path = "./mobilenet.hef"  # 컴파일된 HEF 모델 경로
    video_dir = "./data/val/"  # 테스트 비디오 디렉토리
    
    # Hailo 테스트 실행
    results = test_hailo_model(hef_path, video_dir)
    
    print("\n🎉 Hailo 모델 테스트 완료!")
    print("📊 결과 파일:")
    print("   - hailo_video_results_*.csv (비디오별 결과)")
    print("   - hailo_frame_results_*.csv (프레임별 결과)")
    
    # 비교 분석 (PyTorch 결과가 있는 경우)
    pytorch_video_csv = "pytorch_video_results_20250101_120000.csv"  # 실제 파일명으로 변경
    pytorch_frame_csv = "pytorch_frame_results_20250101_120000.csv"  # 실제 파일명으로 변경
    hailo_video_csv = "hailo_video_results_20250101_120000.csv"      # 실제 파일명으로 변경
    hailo_frame_csv = "hailo_frame_results_20250101_120000.csv"      # 실제 파일명으로 변경
    
    # 파일이 존재하면 비교 분석 실행
    if all(os.path.exists(f) for f in [pytorch_video_csv, hailo_video_csv]):
        compare_models(pytorch_video_csv, pytorch_frame_csv, hailo_video_csv, hailo_frame_csv)
    else:
        print("\n💡 PyTorch 결과와 비교하려면 두 모델을 모두 실행하세요.")
