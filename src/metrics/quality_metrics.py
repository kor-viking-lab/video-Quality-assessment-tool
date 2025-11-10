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

    def calculate_denoise_metrics(
        self,
        original_video: str,
        denoised_video: str,
        num_frames: int = 20,
        remove_letterbox: bool = True,
    ) -> Dict[str, Any]:
        """디노이즈 품질 평가 (원본 노이즈 비디오 vs 디노이즈된 비디오)"""
        try:
            print("원본 비디오에서 프레임 추출 중...")
            original_frames = self.extract_frames(original_video, num_frames)

            print("디노이즈된 비디오에서 프레임 추출 중...")
            denoised_frames = self.extract_frames(denoised_video, num_frames)

            results = {
                "original_video": original_video,
                "denoised_video": denoised_video,
                "frames_analyzed": min(len(original_frames), len(denoised_frames)),
                "letterbox_handling": remove_letterbox,
                "evaluation_type": "denoising",
            }

            # 기본 품질 메트릭 (PSNR, SSIM)
            print("PSNR 계산 중...")
            results["psnr"] = self.calculate_psnr(
                original_frames, denoised_frames, remove_letterbox
            )

            print("SSIM 계산 중...")
            results["ssim"] = self.calculate_ssim(
                original_frames, denoised_frames, remove_letterbox
            )

            # 디노이즈 특화 메트릭
            print("노이즈 제거 효과 분석 중...")
            results["noise_reduction"] = self._calculate_noise_reduction(
                original_frames, denoised_frames, remove_letterbox
            )

            print("디테일 보존도 분석 중...")
            results["detail_preservation"] = self._calculate_detail_preservation(
                original_frames, denoised_frames, remove_letterbox
            )

            print("블러링 정도 분석 중...")
            results["blur_amount"] = self._calculate_blur_amount(
                denoised_frames
            )

            return results

        except Exception as e:
            raise Exception(f"디노이즈 품질 메트릭 계산 중 오류: {str(e)}")

    def _calculate_noise_reduction(
        self,
        original_frames: List[np.ndarray],
        denoised_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """노이즈 제거 효과 측정"""
        noise_reduction_values = []

        for orig_frame, denoised_frame in zip(original_frames, denoised_frames):
            # 레터박스 제거
            if remove_letterbox:
                crop_box = self.detect_content_region(orig_frame)
                orig_frame = self.crop_to_content(orig_frame, crop_box)

                # 비교 프레임 크롭
                orig_h, orig_w = orig_frame.shape[:2]
                denoised_h, denoised_w = denoised_frame.shape[:2]

                scale_x = denoised_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = denoised_h / crop_box[3] if crop_box[3] > 0 else 1

                denoised_x = int(crop_box[0] * scale_x)
                denoised_y = int(crop_box[1] * scale_y)
                denoised_w_crop = int(crop_box[2] * scale_x)
                denoised_h_crop = int(crop_box[3] * scale_y)

                denoised_x = max(0, min(denoised_x, denoised_w - 1))
                denoised_y = max(0, min(denoised_y, denoised_h - 1))
                denoised_w_crop = min(denoised_w_crop, denoised_w - denoised_x)
                denoised_h_crop = min(denoised_h_crop, denoised_h - denoised_y)

                denoised_frame = denoised_frame[
                    denoised_y : denoised_y + denoised_h_crop,
                    denoised_x : denoised_x + denoised_w_crop
                ]

            # 프레임 크기 맞추기
            if orig_frame.shape != denoised_frame.shape:
                denoised_frame = cv2.resize(
                    denoised_frame, (orig_frame.shape[1], orig_frame.shape[0])
                )

            # 그레이스케일 변환
            orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
            denoised_gray = cv2.cvtColor(denoised_frame, cv2.COLOR_RGB2GRAY)

            # 고주파 성분 추출 (노이즈 추정)
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            orig_noise = cv2.filter2D(orig_gray, cv2.CV_64F, kernel)
            denoised_noise = cv2.filter2D(denoised_gray, cv2.CV_64F, kernel)

            # 노이즈 레벨 계산
            orig_noise_level = float(np.std(orig_noise))
            denoised_noise_level = float(np.std(denoised_noise))

            # 노이즈 감소율 계산 (%)
            if orig_noise_level > 0:
                noise_reduction_ratio = (
                    (orig_noise_level - denoised_noise_level) / orig_noise_level * 100
                )
            else:
                noise_reduction_ratio = 0

            noise_reduction_values.append({
                "original_noise_level": orig_noise_level,
                "denoised_noise_level": denoised_noise_level,
                "reduction_ratio": noise_reduction_ratio,
            })

        # 통계 계산
        return {
            "mean_original_noise": float(
                np.mean([nv["original_noise_level"] for nv in noise_reduction_values])
            ),
            "mean_denoised_noise": float(
                np.mean([nv["denoised_noise_level"] for nv in noise_reduction_values])
            ),
            "mean_reduction_ratio": float(
                np.mean([nv["reduction_ratio"] for nv in noise_reduction_values])
            ),
            "frame_values": noise_reduction_values,
        }

    def _calculate_detail_preservation(
        self,
        original_frames: List[np.ndarray],
        denoised_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """디테일 보존도 측정 (엣지 보존)"""
        detail_preservation_values = []

        for orig_frame, denoised_frame in zip(original_frames, denoised_frames):
            # 레터박스 제거
            if remove_letterbox:
                crop_box = self.detect_content_region(orig_frame)
                orig_frame = self.crop_to_content(orig_frame, crop_box)

                orig_h, orig_w = orig_frame.shape[:2]
                denoised_h, denoised_w = denoised_frame.shape[:2]

                scale_x = denoised_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = denoised_h / crop_box[3] if crop_box[3] > 0 else 1

                denoised_x = int(crop_box[0] * scale_x)
                denoised_y = int(crop_box[1] * scale_y)
                denoised_w_crop = int(crop_box[2] * scale_x)
                denoised_h_crop = int(crop_box[3] * scale_y)

                denoised_x = max(0, min(denoised_x, denoised_w - 1))
                denoised_y = max(0, min(denoised_y, denoised_h - 1))
                denoised_w_crop = min(denoised_w_crop, denoised_w - denoised_x)
                denoised_h_crop = min(denoised_h_crop, denoised_h - denoised_y)

                denoised_frame = denoised_frame[
                    denoised_y : denoised_y + denoised_h_crop,
                    denoised_x : denoised_x + denoised_w_crop
                ]

            # 프레임 크기 맞추기
            if orig_frame.shape != denoised_frame.shape:
                denoised_frame = cv2.resize(
                    denoised_frame, (orig_frame.shape[1], orig_frame.shape[0])
                )

            # 그레이스케일 변환
            orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
            denoised_gray = cv2.cvtColor(denoised_frame, cv2.COLOR_RGB2GRAY)

            # Sobel 엣지 검출
            orig_edges_x = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
            orig_edges_y = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
            orig_edges = np.sqrt(orig_edges_x**2 + orig_edges_y**2)

            denoised_edges_x = cv2.Sobel(denoised_gray, cv2.CV_64F, 1, 0, ksize=3)
            denoised_edges_y = cv2.Sobel(denoised_gray, cv2.CV_64F, 0, 1, ksize=3)
            denoised_edges = np.sqrt(denoised_edges_x**2 + denoised_edges_y**2)

            # 엣지 강도 계산
            orig_edge_strength = float(np.mean(orig_edges))
            denoised_edge_strength = float(np.mean(denoised_edges))

            # 디테일 보존율 (%)
            if orig_edge_strength > 0:
                preservation_ratio = (denoised_edge_strength / orig_edge_strength) * 100
            else:
                preservation_ratio = 100.0

            detail_preservation_values.append({
                "original_edge_strength": orig_edge_strength,
                "denoised_edge_strength": denoised_edge_strength,
                "preservation_ratio": preservation_ratio,
            })

        return {
            "mean_preservation_ratio": float(
                np.mean([dv["preservation_ratio"] for dv in detail_preservation_values])
            ),
            "std_preservation_ratio": float(
                np.std([dv["preservation_ratio"] for dv in detail_preservation_values])
            ),
            "frame_values": detail_preservation_values,
        }

    def _calculate_blur_amount(
        self, frames: List[np.ndarray]
    ) -> Dict[str, float]:
        """블러링 정도 측정 (라플라시안 분산)"""
        blur_values = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = float(laplacian.var())
            blur_values.append(blur_score)

        return {
            "mean_blur_score": float(np.mean(blur_values)),
            "std_blur_score": float(np.std(blur_values)),
            "min_blur_score": float(np.min(blur_values)),
            "max_blur_score": float(np.max(blur_values)),
            "frame_values": blur_values,
            "note": "높은 점수는 선명함을, 낮은 점수는 블러를 의미합니다",
        }

    def calculate_colorization_metrics(
        self,
        reference_color_video: str,
        colorized_video: str,
        num_frames: int = 20,
        remove_letterbox: bool = True,
    ) -> Dict[str, Any]:
        """색복원 품질 평가 (원본 컬러 비디오 vs 흑백에서 복원된 컬러 비디오)"""
        try:
            print("원본 컬러 비디오에서 프레임 추출 중...")
            reference_frames = self.extract_frames(reference_color_video, num_frames)

            print("색복원된 비디오에서 프레임 추출 중...")
            colorized_frames = self.extract_frames(colorized_video, num_frames)

            results = {
                "reference_video": reference_color_video,
                "colorized_video": colorized_video,
                "frames_analyzed": min(len(reference_frames), len(colorized_frames)),
                "letterbox_handling": remove_letterbox,
                "evaluation_type": "colorization",
            }

            # 기본 품질 메트릭
            print("PSNR 계산 중...")
            results["psnr"] = self.calculate_psnr(
                reference_frames, colorized_frames, remove_letterbox
            )

            print("SSIM 계산 중...")
            results["ssim"] = self.calculate_ssim(
                reference_frames, colorized_frames, remove_letterbox
            )

            # 색복원 특화 메트릭
            print("색상 정확도 분석 중...")
            results["color_accuracy"] = self._calculate_color_accuracy(
                reference_frames, colorized_frames, remove_letterbox
            )

            print("채도 분석 중...")
            results["saturation_metrics"] = self._calculate_saturation_metrics(
                reference_frames, colorized_frames, remove_letterbox
            )

            print("색조 차이 분석 중...")
            results["hue_difference"] = self._calculate_hue_difference(
                reference_frames, colorized_frames, remove_letterbox
            )

            print("색상 일관성 분석 중...")
            results["color_consistency"] = self._calculate_color_consistency(
                colorized_frames
            )

            return results

        except Exception as e:
            raise Exception(f"색복원 품질 메트릭 계산 중 오류: {str(e)}")

    def _calculate_color_accuracy(
        self,
        reference_frames: List[np.ndarray],
        colorized_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """색상 정확도 측정 (Lab 색공간 기반)"""
        color_accuracy_values = []

        for ref_frame, col_frame in zip(reference_frames, colorized_frames):
            # 레터박스 제거
            if remove_letterbox:
                crop_box = self.detect_content_region(ref_frame)
                ref_frame = self.crop_to_content(ref_frame, crop_box)

                ref_h, ref_w = ref_frame.shape[:2]
                col_h, col_w = col_frame.shape[:2]

                scale_x = col_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = col_h / crop_box[3] if crop_box[3] > 0 else 1

                col_x = int(crop_box[0] * scale_x)
                col_y = int(crop_box[1] * scale_y)
                col_w_crop = int(crop_box[2] * scale_x)
                col_h_crop = int(crop_box[3] * scale_y)

                col_x = max(0, min(col_x, col_w - 1))
                col_y = max(0, min(col_y, col_h - 1))
                col_w_crop = min(col_w_crop, col_w - col_x)
                col_h_crop = min(col_h_crop, col_h - col_y)

                col_frame = col_frame[
                    col_y : col_y + col_h_crop, col_x : col_x + col_w_crop
                ]

            # 프레임 크기 맞추기
            if ref_frame.shape != col_frame.shape:
                col_frame = cv2.resize(
                    col_frame, (ref_frame.shape[1], ref_frame.shape[0])
                )

            # RGB를 Lab 색공간으로 변환 (지각적 색상 차이 측정에 더 적합)
            ref_lab = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2LAB).astype(np.float32)
            col_lab = cv2.cvtColor(col_frame, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Delta E (색상 차이) 계산
            delta_e = np.sqrt(np.sum((ref_lab - col_lab) ** 2, axis=2))
            mean_delta_e = float(np.mean(delta_e))

            # 채널별 차이
            l_diff = float(np.mean(np.abs(ref_lab[:, :, 0] - col_lab[:, :, 0])))
            a_diff = float(np.mean(np.abs(ref_lab[:, :, 1] - col_lab[:, :, 1])))
            b_diff = float(np.mean(np.abs(ref_lab[:, :, 2] - col_lab[:, :, 2])))

            color_accuracy_values.append({
                "delta_e": mean_delta_e,
                "l_difference": l_diff,  # 밝기 차이
                "a_difference": a_diff,  # 초록-빨강 차이
                "b_difference": b_diff,  # 파랑-노랑 차이
            })

        return {
            "mean_delta_e": float(
                np.mean([ca["delta_e"] for ca in color_accuracy_values])
            ),
            "std_delta_e": float(
                np.std([ca["delta_e"] for ca in color_accuracy_values])
            ),
            "mean_l_diff": float(
                np.mean([ca["l_difference"] for ca in color_accuracy_values])
            ),
            "mean_a_diff": float(
                np.mean([ca["a_difference"] for ca in color_accuracy_values])
            ),
            "mean_b_diff": float(
                np.mean([ca["b_difference"] for ca in color_accuracy_values])
            ),
            "frame_values": color_accuracy_values,
            "note": "낮은 Delta E 값은 더 정확한 색상을 의미합니다 (0=완벽, <2.3=지각 불가)",
        }

    def _calculate_saturation_metrics(
        self,
        reference_frames: List[np.ndarray],
        colorized_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """채도 메트릭 계산 (HSV 기반)"""
        saturation_values = []

        for ref_frame, col_frame in zip(reference_frames, colorized_frames):
            # 레터박스 제거
            if remove_letterbox:
                crop_box = self.detect_content_region(ref_frame)
                ref_frame = self.crop_to_content(ref_frame, crop_box)

                ref_h, ref_w = ref_frame.shape[:2]
                col_h, col_w = col_frame.shape[:2]

                scale_x = col_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = col_h / crop_box[3] if crop_box[3] > 0 else 1

                col_x = int(crop_box[0] * scale_x)
                col_y = int(crop_box[1] * scale_y)
                col_w_crop = int(crop_box[2] * scale_x)
                col_h_crop = int(crop_box[3] * scale_y)

                col_x = max(0, min(col_x, col_w - 1))
                col_y = max(0, min(col_y, col_h - 1))
                col_w_crop = min(col_w_crop, col_w - col_x)
                col_h_crop = min(col_h_crop, col_h - col_y)

                col_frame = col_frame[
                    col_y : col_y + col_h_crop, col_x : col_x + col_w_crop
                ]

            # 프레임 크기 맞추기
            if ref_frame.shape != col_frame.shape:
                col_frame = cv2.resize(
                    col_frame, (ref_frame.shape[1], ref_frame.shape[0])
                )

            # HSV 색공간으로 변환
            ref_hsv = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2HSV)
            col_hsv = cv2.cvtColor(col_frame, cv2.COLOR_RGB2HSV)

            # 채도 (S 채널) 분석
            ref_saturation = float(np.mean(ref_hsv[:, :, 1]))
            col_saturation = float(np.mean(col_hsv[:, :, 1]))

            # 채도 차이
            saturation_diff = float(np.mean(np.abs(
                ref_hsv[:, :, 1].astype(np.float32) - col_hsv[:, :, 1].astype(np.float32)
            )))

            saturation_values.append({
                "reference_saturation": ref_saturation,
                "colorized_saturation": col_saturation,
                "saturation_difference": saturation_diff,
            })

        return {
            "mean_ref_saturation": float(
                np.mean([sv["reference_saturation"] for sv in saturation_values])
            ),
            "mean_col_saturation": float(
                np.mean([sv["colorized_saturation"] for sv in saturation_values])
            ),
            "mean_saturation_diff": float(
                np.mean([sv["saturation_difference"] for sv in saturation_values])
            ),
            "std_saturation_diff": float(
                np.std([sv["saturation_difference"] for sv in saturation_values])
            ),
            "frame_values": saturation_values,
        }

    def _calculate_hue_difference(
        self,
        reference_frames: List[np.ndarray],
        colorized_frames: List[np.ndarray],
        remove_letterbox: bool = True,
    ) -> Dict[str, float]:
        """색조(Hue) 차이 측정"""
        hue_diff_values = []

        for ref_frame, col_frame in zip(reference_frames, colorized_frames):
            # 레터박스 제거
            if remove_letterbox:
                crop_box = self.detect_content_region(ref_frame)
                ref_frame = self.crop_to_content(ref_frame, crop_box)

                ref_h, ref_w = ref_frame.shape[:2]
                col_h, col_w = col_frame.shape[:2]

                scale_x = col_w / crop_box[2] if crop_box[2] > 0 else 1
                scale_y = col_h / crop_box[3] if crop_box[3] > 0 else 1

                col_x = int(crop_box[0] * scale_x)
                col_y = int(crop_box[1] * scale_y)
                col_w_crop = int(crop_box[2] * scale_x)
                col_h_crop = int(crop_box[3] * scale_y)

                col_x = max(0, min(col_x, col_w - 1))
                col_y = max(0, min(col_y, col_h - 1))
                col_w_crop = min(col_w_crop, col_w - col_x)
                col_h_crop = min(col_h_crop, col_h - col_y)

                col_frame = col_frame[
                    col_y : col_y + col_h_crop, col_x : col_x + col_w_crop
                ]

            # 프레임 크기 맞추기
            if ref_frame.shape != col_frame.shape:
                col_frame = cv2.resize(
                    col_frame, (ref_frame.shape[1], ref_frame.shape[0])
                )

            # HSV 색공간으로 변환
            ref_hsv = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2HSV)
            col_hsv = cv2.cvtColor(col_frame, cv2.COLOR_RGB2HSV)

            # Hue는 순환형이므로 각도 차이 계산 (0-180 범위)
            ref_hue = ref_hsv[:, :, 0].astype(np.float32)
            col_hue = col_hsv[:, :, 0].astype(np.float32)

            # 순환 거리 계산
            hue_diff = np.minimum(
                np.abs(ref_hue - col_hue),
                180 - np.abs(ref_hue - col_hue)
            )

            mean_hue_diff = float(np.mean(hue_diff))

            hue_diff_values.append(mean_hue_diff)

        return {
            "mean_hue_difference": float(np.mean(hue_diff_values)),
            "std_hue_difference": float(np.std(hue_diff_values)),
            "min_hue_difference": float(np.min(hue_diff_values)),
            "max_hue_difference": float(np.max(hue_diff_values)),
            "frame_values": hue_diff_values,
            "note": "낮은 값은 더 정확한 색조를 의미합니다 (0-180 범위)",
        }

    def _calculate_color_consistency(
        self, frames: List[np.ndarray]
    ) -> Dict[str, float]:
        """색상 일관성 분석 (프레임 간 색상 변화)"""
        if len(frames) < 2:
            return {
                "temporal_consistency": 0.0,
                "note": "프레임 수가 부족하여 일관성을 측정할 수 없습니다",
            }

        consistency_scores = []

        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]

            # Lab 색공간으로 변환
            lab1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2LAB).astype(np.float32)

            # 프레임 간 색상 차이
            color_diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
            mean_diff = float(np.mean(color_diff))

            consistency_scores.append(mean_diff)

        return {
            "mean_temporal_difference": float(np.mean(consistency_scores)),
            "std_temporal_difference": float(np.std(consistency_scores)),
            "frame_to_frame_differences": consistency_scores,
            "note": "낮은 값은 프레임 간 색상이 일관적임을 의미합니다",
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

        # 디노이즈 특화 해석
        if "noise_reduction" in results:
            reduction_ratio = results["noise_reduction"]["mean_reduction_ratio"]
            if reduction_ratio >= 70:
                interpretation["noise_reduction"] = "매우 효과적인 노이즈 제거"
            elif reduction_ratio >= 50:
                interpretation["noise_reduction"] = "효과적인 노이즈 제거"
            elif reduction_ratio >= 30:
                interpretation["noise_reduction"] = "보통 수준의 노이즈 제거"
            else:
                interpretation["noise_reduction"] = "낮은 노이즈 제거 효과"

        if "detail_preservation" in results:
            preservation = results["detail_preservation"]["mean_preservation_ratio"]
            if preservation >= 90:
                interpretation["detail_preservation"] = "매우 우수한 디테일 보존"
            elif preservation >= 75:
                interpretation["detail_preservation"] = "우수한 디테일 보존"
            elif preservation >= 60:
                interpretation["detail_preservation"] = "보통 수준의 디테일 보존"
            else:
                interpretation["detail_preservation"] = "디테일 손실 발생"

        if "blur_amount" in results:
            blur_score = results["blur_amount"]["mean_blur_score"]
            if blur_score >= 500:
                interpretation["blur_amount"] = "매우 선명함"
            elif blur_score >= 100:
                interpretation["blur_amount"] = "선명함"
            elif blur_score >= 50:
                interpretation["blur_amount"] = "약간 블러됨"
            else:
                interpretation["blur_amount"] = "심하게 블러됨"

        # 색복원 특화 해석
        if "color_accuracy" in results:
            delta_e = results["color_accuracy"]["mean_delta_e"]
            if delta_e < 2.3:
                interpretation["color_accuracy"] = "지각적으로 완벽한 색상"
            elif delta_e < 5:
                interpretation["color_accuracy"] = "매우 정확한 색상"
            elif delta_e < 10:
                interpretation["color_accuracy"] = "정확한 색상"
            elif delta_e < 20:
                interpretation["color_accuracy"] = "보통 수준의 색상 정확도"
            else:
                interpretation["color_accuracy"] = "색상 차이가 큼"

        if "hue_difference" in results:
            hue_diff = results["hue_difference"]["mean_hue_difference"]
            if hue_diff < 5:
                interpretation["hue_difference"] = "매우 정확한 색조"
            elif hue_diff < 10:
                interpretation["hue_difference"] = "정확한 색조"
            elif hue_diff < 20:
                interpretation["hue_difference"] = "보통 수준의 색조 정확도"
            else:
                interpretation["hue_difference"] = "색조 차이가 큼"

        if "color_consistency" in results:
            temporal_diff = results["color_consistency"]["mean_temporal_difference"]
            if temporal_diff < 5:
                interpretation["color_consistency"] = "매우 일관된 색상"
            elif temporal_diff < 10:
                interpretation["color_consistency"] = "일관된 색상"
            elif temporal_diff < 20:
                interpretation["color_consistency"] = "보통 수준의 색상 일관성"
            else:
                interpretation["color_consistency"] = "색상 일관성 부족"

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

        # 평가 타입 확인
        eval_type = results.get("evaluation_type", "")

        # 파일 정보 출력
        if eval_type == "denoising":
            print(f"원본 비디오: {Path(results['original_video']).name}")
            print(f"디노이즈 비디오: {Path(results['denoised_video']).name}")
        elif eval_type == "colorization":
            print(f"참조 컬러 비디오: {Path(results['reference_video']).name}")
            print(f"색복원 비디오: {Path(results['colorized_video']).name}")
        elif "reference_video" in results:
            print(f"참조 비디오: {Path(results['reference_video']).name}")
            if "distorted_video" in results:
                print(f"비교 비디오: {Path(results['distorted_video']).name}")
            elif "colorized_video" in results:
                print(f"색복원 비디오: {Path(results['colorized_video']).name}")
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

        # 디노이즈 특화 메트릭
        if "noise_reduction" in results:
            nr = results["noise_reduction"]
            print(f"\n노이즈 제거 효과:")
            print(f"  원본 노이즈 레벨: {nr['mean_original_noise']:.2f}")
            print(f"  디노이즈 후 노이즈 레벨: {nr['mean_denoised_noise']:.2f}")
            print(f"  노이즈 감소율: {nr['mean_reduction_ratio']:.2f}%")

        if "detail_preservation" in results:
            dp = results["detail_preservation"]
            print(f"\n디테일 보존도:")
            print(f"  평균 보존율: {dp['mean_preservation_ratio']:.2f}%")
            print(f"  표준편차: {dp['std_preservation_ratio']:.2f}%")

        if "blur_amount" in results:
            blur = results["blur_amount"]
            print(f"\n선명도 분석:")
            print(f"  블러 점수: {blur['mean_blur_score']:.2f}")
            print(f"  ({blur['note']})")

        # 색복원 특화 메트릭
        if "color_accuracy" in results:
            ca = results["color_accuracy"]
            print(f"\n색상 정확도 (Delta E):")
            print(f"  평균 Delta E: {ca['mean_delta_e']:.2f}")
            print(f"  표준편차: {ca['std_delta_e']:.2f}")
            print(f"  밝기(L) 차이: {ca['mean_l_diff']:.2f}")
            print(f"  초록-빨강(a) 차이: {ca['mean_a_diff']:.2f}")
            print(f"  파랑-노랑(b) 차이: {ca['mean_b_diff']:.2f}")
            print(f"  ({ca['note']})")

        if "saturation_metrics" in results:
            sm = results["saturation_metrics"]
            print(f"\n채도 분석:")
            print(f"  참조 평균 채도: {sm['mean_ref_saturation']:.2f}")
            print(f"  복원 평균 채도: {sm['mean_col_saturation']:.2f}")
            print(f"  채도 차이: {sm['mean_saturation_diff']:.2f}")

        if "hue_difference" in results:
            hd = results["hue_difference"]
            print(f"\n색조 차이:")
            print(f"  평균: {hd['mean_hue_difference']:.2f}°")
            print(f"  표준편차: {hd['std_hue_difference']:.2f}°")
            print(f"  ({hd['note']})")

        if "color_consistency" in results:
            cc = results["color_consistency"]
            print(f"\n색상 일관성 (시간적):")
            if "mean_temporal_difference" in cc:
                print(f"  프레임 간 평균 차이: {cc['mean_temporal_difference']:.2f}")
                print(f"  표준편차: {cc['std_temporal_difference']:.2f}")
            print(f"  ({cc['note']})")

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
