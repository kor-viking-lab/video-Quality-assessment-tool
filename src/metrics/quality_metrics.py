import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import ffmpeg


class QualityMetricsCalculator:
    """
    비디오 품질 메트릭을 계산하는 클래스
    PSNR, SSIM, VMAF 등의 품질 지표를 측정합니다.
    """

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vmaf_model_path = self._find_vmaf_model()

    def _find_vmaf_model(self) -> Optional[str]:
        """VMAF 모델 파일 경로 찾기"""
        # 프로젝트 기준 경로들
        base_dir = Path(__file__).parent.parent.parent  # 프로젝트 루트 디렉토리

        possible_paths = [
            base_dir / "vmaf_models" / "vmaf_v0.6.1.json",
            base_dir / "vmaf_models" / "vmaf_4k_v0.6.1.json",
            base_dir / "vmaf" / "model" / "vmaf_v0.6.1.json",
            "./vmaf_models/vmaf_v0.6.1.json",
            "C:/ffmpeg/bin/vmaf_v0.6.1.json",
        ]

        for path in possible_paths:
            if os.path.exists(str(path)):
                print(f"VMAF 모델 발견: {path}")
                return str(path)

        print("VMAF 모델을 찾을 수 없습니다. VMAF 계산이 제한될 수 있습니다.")
        return None

    def extract_frames(
        self, video_path: str, num_frames: int = 10, start_time: float = 0
    ) -> List[np.ndarray]:
        """비디오에서 프레임 추출"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

            # 비디오 정보 가져오기
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 시작 프레임 설정
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # 프레임 간격 계산
            frame_interval = max(1, (total_frames - start_frame) // num_frames)

            frames = []
            frame_count = 0

            while len(frames) < num_frames and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # BGR을 RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                frame_count += 1

            cap.release()
            return frames

        except Exception as e:
            raise Exception(f"프레임 추출 중 오류: {str(e)}")

    def detect_content_region(
        self, frame: np.ndarray, threshold: int = 10
    ) -> Tuple[int, int, int, int]:
        """프레임에서 실제 컨텐츠 영역 검출 (블랙 바 제거)"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 블랙 바가 아닌 영역 찾기 (threshold보다 밝은 픽셀)
        mask = gray > threshold

        # 컨텐츠가 있는 행/열 찾기
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # 컨텐츠를 찾을 수 없으면 전체 프레임 반환
            return 0, 0, frame.shape[1], frame.shape[0]

        # 경계 좌표 계산
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

    def crop_to_content(
        self, frame: np.ndarray, crop_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """프레임을 컨텐츠 영역으로 크롭"""
        x, y, w, h = crop_box
        return frame[y : y + h, x : x + w]

    def calculate_psnr(
        self,
        reference_frames: List[np.ndarray],
        distorted_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """PSNR (Peak Signal-to-Noise Ratio) 계산"""
        if len(reference_frames) != len(distorted_frames):
            raise ValueError("참조 프레임과 비교 프레임 수가 일치하지 않습니다.")

        psnr_values = []

        for ref_frame, dist_frame in zip(reference_frames, distorted_frames):
            # 레터박스 제거 옵션
            if remove_letterbox:
                # 참조 프레임에서 컨텐츠 영역 검출
                crop_box = self.detect_content_region(ref_frame)
                ref_frame = self.crop_to_content(ref_frame, crop_box)

                # 비교 프레임도 같은 영역으로 크롭 (비율 조정)
                ref_h, ref_w = ref_frame.shape[:2]
                dist_h, dist_w = dist_frame.shape[:2]

                # 스케일 비율 계산
                scale_x = dist_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = dist_h / crop_box[3] if crop_box[3] > 0 else 1

                # 비교 프레임의 해당 영역 크롭
                dist_x = int(crop_box[0] * scale_x)
                dist_y = int(crop_box[1] * scale_y)
                dist_w_crop = int(crop_box[2] * scale_x)
                dist_h_crop = int(crop_box[3] * scale_y)

                # 경계 체크
                dist_x = max(0, min(dist_x, dist_w - 1))
                dist_y = max(0, min(dist_y, dist_h - 1))
                dist_w_crop = min(dist_w_crop, dist_w - dist_x)
                dist_h_crop = min(dist_h_crop, dist_h - dist_y)

                dist_frame = dist_frame[
                    dist_y : dist_y + dist_h_crop, dist_x : dist_x + dist_w_crop
                ]

            # 프레임 크기가 다르면 리사이즈
            if ref_frame.shape != dist_frame.shape:
                dist_frame = cv2.resize(
                    dist_frame, (ref_frame.shape[1], ref_frame.shape[0])
                )

            # PSNR 계산 (scikit-image 사용)
            psnr_value = psnr(ref_frame, dist_frame, data_range=255)
            psnr_values.append(psnr_value)

        return {
            "mean_psnr": float(np.mean(psnr_values)),
            "min_psnr": float(np.min(psnr_values)),
            "max_psnr": float(np.max(psnr_values)),
            "std_psnr": float(np.std(psnr_values)),
            "frame_psnr_values": [float(x) for x in psnr_values],
            "letterbox_removed": remove_letterbox,
        }

    def calculate_ssim(
        self,
        reference_frames: List[np.ndarray],
        distorted_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """SSIM (Structural Similarity Index) 계산"""
        if len(reference_frames) != len(distorted_frames):
            raise ValueError("참조 프레임과 비교 프레임 수가 일치하지 않습니다.")

        ssim_values = []

        for ref_frame, dist_frame in zip(reference_frames, distorted_frames):
            # 레터박스 제거 옵션
            if remove_letterbox:
                # 참조 프레임에서 컨텐츠 영역 검출
                crop_box = self.detect_content_region(ref_frame)
                ref_frame = self.crop_to_content(ref_frame, crop_box)

                # 비교 프레임도 같은 비율로 크롭
                ref_h, ref_w = ref_frame.shape[:2]
                dist_h, dist_w = dist_frame.shape[:2]

                scale_x = dist_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = dist_h / crop_box[3] if crop_box[3] > 0 else 1

                dist_x = int(crop_box[0] * scale_x)
                dist_y = int(crop_box[1] * scale_y)
                dist_w_crop = int(crop_box[2] * scale_x)
                dist_h_crop = int(crop_box[3] * scale_y)

                dist_x = max(0, min(dist_x, dist_w - 1))
                dist_y = max(0, min(dist_y, dist_h - 1))
                dist_w_crop = min(dist_w_crop, dist_w - dist_x)
                dist_h_crop = min(dist_h_crop, dist_h - dist_y)

                dist_frame = dist_frame[
                    dist_y : dist_y + dist_h_crop, dist_x : dist_x + dist_w_crop
                ]

            # 프레임 크기가 다르면 리사이즈
            if ref_frame.shape != dist_frame.shape:
                dist_frame = cv2.resize(
                    dist_frame, (ref_frame.shape[1], ref_frame.shape[0])
                )

            # 그레이스케일로 변환
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
            dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_RGB2GRAY)

            # SSIM 계산
            ssim_value = ssim(ref_gray, dist_gray, data_range=255)
            ssim_values.append(ssim_value)

        return {
            "mean_ssim": float(np.mean(ssim_values)),
            "min_ssim": float(np.min(ssim_values)),
            "max_ssim": float(np.max(ssim_values)),
            "std_ssim": float(np.std(ssim_values)),
            "frame_ssim_values": [float(x) for x in ssim_values],
            "letterbox_removed": remove_letterbox,
        }

    def calculate_vmaf_ffmpeg(
        self, reference_video: str, distorted_video: str
    ) -> Dict[str, Any]:
        """FFmpeg을 사용한 VMAF 계산 (동일 해상도만 지원)"""
        try:
            # 비디오 해상도 정보 확인
            ref_probe = ffmpeg.probe(reference_video)
            dist_probe = ffmpeg.probe(distorted_video)

            ref_stream = next(
                s for s in ref_probe["streams"] if s["codec_type"] == "video"
            )
            dist_stream = next(
                s for s in dist_probe["streams"] if s["codec_type"] == "video"
            )

            ref_width = int(ref_stream["width"])
            ref_height = int(ref_stream["height"])
            dist_width = int(dist_stream["width"])
            dist_height = int(dist_stream["height"])

            print(f"참조 비디오 해상도: {ref_width}x{ref_height}")
            print(f"비교 비디오 해상도: {dist_width}x{dist_height}")

            # 해상도가 다르면 VMAF 계산 건너뛰기
            if ref_width != dist_width or ref_height != dist_height:
                return {
                    "status": "skipped",
                    "reason": "VMAF 기능은 동일한 해상도가 아닙니다.",
                    "resolution_info": {
                        "reference": f"{ref_width}x{ref_height}",
                        "distorted": f"{dist_width}x{dist_height}",
                    },
                    "note": "VMAF는 동일한 해상도의 비디오만 비교할 수 있습니다.",
                }

            # 해상도가 같을 때만 VMAF 계산 진행
            print("해상도가 일치합니다. VMAF 계산을 진행합니다.")

            # 임시 출력 파일
            output_file = os.path.join(self.temp_dir, "vmaf_output.json")

            # FFmpeg VMAF 필터 구성
            if self.vmaf_model_path and os.path.exists(self.vmaf_model_path):
                vmaf_filter = f"libvmaf=model_path={self.vmaf_model_path}:log_path={output_file}:log_fmt=json"
            else:
                # 기본 VMAF 모델 사용 (FFmpeg 내장)
                vmaf_filter = f"libvmaf=log_path={output_file}:log_fmt=json"

            # FFmpeg 명령어 실행
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                distorted_video,
                "-i",
                reference_video,
                "-lavfi",
                vmaf_filter,
                "-f",
                "null",
                "-y",
                "-",
            ]

            print("VMAF 계산 중... (시간이 걸릴 수 있습니다)")

            # Windows 인코딩 문제 해결
            import locale

            encoding = locale.getpreferredencoding()

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    encoding="utf-8",
                    errors="ignore",
                )
            except UnicodeDecodeError:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    encoding=encoding,
                    errors="ignore",
                )

            if result.returncode != 0:
                try:
                    stderr_msg = result.stderr if result.stderr else "알 수 없는 오류"
                except:
                    stderr_msg = "인코딩 오류로 에러 메시지를 읽을 수 없습니다"

                raise Exception(f"FFmpeg VMAF 계산 실패: {stderr_msg}")

            # 결과 파일 읽기
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        vmaf_data = json.load(f)
                except UnicodeDecodeError:
                    with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
                        vmaf_data = json.load(f)

                # VMAF 점수 추출
                frames_data = vmaf_data.get("frames", [])
                vmaf_scores = []

                for frame in frames_data:
                    metrics = frame.get("metrics", {})
                    if "vmaf" in metrics:
                        vmaf_scores.append(metrics["vmaf"])
                    elif "VMAF_score" in metrics:
                        vmaf_scores.append(metrics["VMAF_score"])

                if vmaf_scores:
                    return {
                        "status": "success",
                        "mean_vmaf": float(np.mean(vmaf_scores)),
                        "min_vmaf": float(np.min(vmaf_scores)),
                        "max_vmaf": float(np.max(vmaf_scores)),
                        "std_vmaf": float(np.std(vmaf_scores)),
                        "frame_count": len(vmaf_scores),
                        "frame_vmaf_values": [float(x) for x in vmaf_scores[:50]],
                        "resolution_info": {
                            "reference": f"{ref_width}x{ref_height}",
                            "distorted": f"{dist_width}x{dist_height}",
                        },
                    }
                else:
                    raise Exception("VMAF 점수를 추출할 수 없습니다.")
            else:
                raise Exception("VMAF 출력 파일을 찾을 수 없습니다.")

        except subprocess.TimeoutExpired:
            raise Exception("VMAF 계산 시간 초과 (5분)")
        except Exception as e:
            raise Exception(f"VMAF 계산 중 오류: {str(e)}")

    def calculate_quality_metrics_comparison(
        self,
        reference_video: str,
        distorted_video: str,
        num_frames: int = 20,
        remove_letterbox: bool = True,
    ) -> Dict[str, Any]:
        """두 비디오 간의 품질 메트릭 비교 계산"""
        try:
            print("참조 비디오에서 프레임 추출 중...")
            ref_frames = self.extract_frames(reference_video, num_frames)

            print("비교 비디오에서 프레임 추출 중...")
            dist_frames = self.extract_frames(distorted_video, num_frames)

            results = {
                "reference_video": reference_video,
                "distorted_video": distorted_video,
                "frames_analyzed": min(len(ref_frames), len(dist_frames)),
                "letterbox_handling": remove_letterbox,
            }

            # 레터박스 정보 출력
            if remove_letterbox:
                print("레터박스/필러박스 자동 제거 모드")
                crop_box = self.detect_content_region(ref_frames[0])
                print(
                    f"검출된 컨텐츠 영역: {crop_box[2]}x{crop_box[3]} (x:{crop_box[0]}, y:{crop_box[1]})"
                )

            # PSNR 계산
            print("PSNR 계산 중...")
            results["psnr"] = self.calculate_psnr(
                ref_frames, dist_frames, remove_letterbox
            )

            # SSIM 계산
            print("SSIM 계산 중...")
            results["ssim"] = self.calculate_ssim(
                ref_frames, dist_frames, remove_letterbox
            )

            # VMAF 계산 (선택사항 - 시간이 오래 걸림)
            try:
                print("VMAF 계산 중...")
                results["vmaf"] = self.calculate_vmaf_ffmpeg(
                    reference_video, distorted_video
                )
            except Exception as e:
                print(f"VMAF 계산 실패: {str(e)}")
                results["vmaf"] = {"error": str(e)}

            return results

        except Exception as e:
            raise Exception(f"품질 메트릭 계산 중 오류: {str(e)}")

    def calculate_single_video_metrics(
        self, video_path: str, num_frames: int = 20
    ) -> Dict[str, Any]:
        """단일 비디오의 품질 지표 계산 (절대적 품질)"""
        try:
            print("비디오에서 프레임 추출 중...")
            frames = self.extract_frames(video_path, num_frames)

            results = {
                "video_path": video_path,
                "frames_analyzed": len(frames),
                "quality_analysis": {},
            }

            # 프레임별 품질 분석
            frame_qualities = []

            for i, frame in enumerate(frames):
                # 프레임 품질 지표들
                frame_quality = self._analyze_frame_quality(frame)
                frame_qualities.append(frame_quality)

            # 통계 계산
            results["quality_analysis"] = {
                "brightness": {
                    "mean": float(
                        np.mean([fq["brightness"] for fq in frame_qualities])
                    ),
                    "std": float(np.std([fq["brightness"] for fq in frame_qualities])),
                },
                "contrast": {
                    "mean": float(np.mean([fq["contrast"] for fq in frame_qualities])),
                    "std": float(np.std([fq["contrast"] for fq in frame_qualities])),
                },
                "sharpness": {
                    "mean": float(np.mean([fq["sharpness"] for fq in frame_qualities])),
                    "std": float(np.std([fq["sharpness"] for fq in frame_qualities])),
                },
                "noise_estimate": {
                    "mean": float(
                        np.mean([fq["noise_estimate"] for fq in frame_qualities])
                    ),
                    "std": float(
                        np.std([fq["noise_estimate"] for fq in frame_qualities])
                    ),
                },
            }

            return results

        except Exception as e:
            raise Exception(f"단일 비디오 품질 분석 중 오류: {str(e)}")

    def _analyze_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """개별 프레임의 품질 분석"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 밝기 (평균 픽셀 값)
        brightness = float(np.mean(gray))

        # 대비 (표준편차)
        contrast = float(np.std(gray))

        # 선명도 (라플라시안 분산)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # 노이즈 추정 (고주파 성분)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_map = cv2.filter2D(gray, cv2.CV_64F, kernel)
        noise_estimate = float(np.std(noise_map))

        return {
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "noise_estimate": noise_estimate,
        }

    def interpret_metrics(self, results: Dict[str, Any]) -> Dict[str, str]:
        """메트릭 결과 해석"""
        interpretation = {}

        # PSNR 해석
        if "psnr" in results:
            mean_psnr = results["psnr"]["mean_psnr"]
            if mean_psnr >= 40:
                interpretation["psnr"] = "매우 우수한 품질"
            elif mean_psnr >= 30:
                interpretation["psnr"] = "우수한 품질"
            elif mean_psnr >= 20:
                interpretation["psnr"] = "보통 품질"
            else:
                interpretation["psnr"] = "낮은 품질"

        # SSIM 해석
        if "ssim" in results:
            mean_ssim = results["ssim"]["mean_ssim"]
            if mean_ssim >= 0.95:
                interpretation["ssim"] = "매우 유사함"
            elif mean_ssim >= 0.8:
                interpretation["ssim"] = "유사함"
            elif mean_ssim >= 0.6:
                interpretation["ssim"] = "보통 유사함"
            else:
                interpretation["ssim"] = "유사하지 않음"

        # VMAF 해석
        if "vmaf" in results:
            vmaf_data = results["vmaf"]
            if vmaf_data.get("status") == "success" and "mean_vmaf" in vmaf_data:
                mean_vmaf = vmaf_data["mean_vmaf"]
                if mean_vmaf >= 90:
                    interpretation["vmaf"] = "매우 우수한 지각적 품질"
                elif mean_vmaf >= 70:
                    interpretation["vmaf"] = "우수한 지각적 품질"
                elif mean_vmaf >= 50:
                    interpretation["vmaf"] = "보통 지각적 품질"
                else:
                    interpretation["vmaf"] = "낮은 지각적 품질"
            elif vmaf_data.get("status") == "skipped":
                interpretation["vmaf"] = "해상도 불일치로 건너뜀"

        return interpretation

    def save_results(self, results: Dict[str, Any], output_path: str):
        """결과를 JSON 파일로 저장"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"품질 메트릭 결과가 저장되었습니다: {output_path}")
        except Exception as e:
            raise Exception(f"결과 저장 중 오류: {str(e)}")

    def print_results(self, results: Dict[str, Any]):
        """결과를 콘솔에 출력"""
        print("\n=== 비디오 품질 메트릭 분석 결과 ===")

        if "reference_video" in results:
            print(f"참조 비디오: {Path(results['reference_video']).name}")
            print(f"비교 비디오: {Path(results['distorted_video']).name}")
        else:
            print(f"분석 비디오: {Path(results['video_path']).name}")

        print(f"분석된 프레임 수: {results.get('frames_analyzed', 0)}")

        # PSNR 결과
        if "psnr" in results:
            psnr_data = results["psnr"]
            print(f"\nPSNR (Peak Signal-to-Noise Ratio):")
            print(f"  평균: {psnr_data['mean_psnr']:.2f} dB")
            print(f"  최소: {psnr_data['min_psnr']:.2f} dB")
            print(f"  최대: {psnr_data['max_psnr']:.2f} dB")

        # SSIM 결과
        if "ssim" in results:
            ssim_data = results["ssim"]
            print(f"\nSSIM (Structural Similarity Index):")
            print(f"  평균: {ssim_data['mean_ssim']:.4f}")
            print(f"  최소: {ssim_data['min_ssim']:.4f}")
            print(f"  최대: {ssim_data['max_ssim']:.4f}")

        # VMAF 결과
        if "vmaf" in results:
            vmaf_data = results["vmaf"]
            if vmaf_data.get("status") == "skipped":
                print(f"\nVMAF: {vmaf_data['reason']}")
                print(f"  참조 해상도: {vmaf_data['resolution_info']['reference']}")
                print(f"  비교 해상도: {vmaf_data['resolution_info']['distorted']}")
                print(f"  참고: {vmaf_data['note']}")
            elif vmaf_data.get("status") == "success":
                print(f"\nVMAF (Video Multi-method Assessment Fusion):")
                print(f"  평균: {vmaf_data['mean_vmaf']:.2f}")
                print(f"  최소: {vmaf_data['min_vmaf']:.2f}")
                print(f"  최대: {vmaf_data['max_vmaf']:.2f}")
            elif "error" in vmaf_data:
                print(f"\nVMAF: 계산 실패 - {vmaf_data['error']}")
            else:
                print(f"\nVMAF: 알 수 없는 상태")

        # 품질 분석 (단일 비디오)
        if "quality_analysis" in results:
            qa = results["quality_analysis"]
            print(f"\n품질 분석:")
            print(f"  밝기: {qa['brightness']['mean']:.1f}")
            print(f"  대비: {qa['contrast']['mean']:.1f}")
            print(f"  선명도: {qa['sharpness']['mean']:.1f}")
            print(f"  노이즈: {qa['noise_estimate']['mean']:.2f}")

        # 해석
        interpretation = self.interpret_metrics(results)
        if interpretation:
            print(f"\n품질 해석:")
            for metric, desc in interpretation.items():
                print(f"  {metric.upper()}: {desc}")


def main():
    """테스트용 메인 함수"""
    calculator = QualityMetricsCalculator()

    print("비디오 품질 메트릭 계산기")
    print("1. 두 비디오 비교 분석")
    print("2. 단일 비디오 품질 분석")

    choice = input("선택하세요 (1 또는 2): ").strip()

    try:
        if choice == "1":
            ref_video = input("참조 비디오 파일 경로: ").strip()
            dist_video = input("비교할 비디오 파일 경로: ").strip()

            if not ref_video or not dist_video:
                print("비디오 파일 경로를 입력해주세요.")
                return

            results = calculator.calculate_quality_metrics_comparison(
                ref_video, dist_video
            )
            calculator.print_results(results)

            # 결과 저장
            output_file = f"quality_comparison_{Path(dist_video).stem}.json"
            calculator.save_results(results, output_file)

        elif choice == "2":
            video_path = input("분석할 비디오 파일 경로: ").strip()

            if not video_path:
                print("비디오 파일 경로를 입력해주세요.")
                return

            results = calculator.calculate_single_video_metrics(video_path)
            calculator.print_results(results)

            # 결과 저장
            output_file = f"quality_analysis_{Path(video_path).stem}.json"
            calculator.save_results(results, output_file)

        else:
            print("잘못된 선택입니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
