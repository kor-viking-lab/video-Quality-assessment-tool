#!/usr/bin/env python3
"""
OTT/DCI 품질평가 도구 - 메인 통합 실행 파일
META(PQL) / NETFLIX(VMAF) 기준 종합 품질 분석 도구

사용 가능한 기능:
1. 비디오 메타데이터 분석
2. 품질 메트릭 비교 (PSNR, SSIM, VMAF)
3. DCI/OTT 표준 준수 검사
4. 종합 품질 분석 (모든 기능 통합)
"""

import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

try:
    from analyzers.video_metadata_analyzer import VideoMetadataAnalyzer
    from metrics.quality_metrics import QualityMetricsCalculator
    from reports.dci_ott_standards_checker import DCIOTTStandardsChecker
    from reports.html_report_generator import HTMLReportGenerator
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("필요한 파일들이 올바른 위치에 있는지 확인하세요:")
    print("- src/analyzers/video_metadata_analyzer.py")
    print("- src/metrics/quality_metrics.py")
    print("- src/reports/dci_ott_standards_checker.py")
    sys.exit(1)


class OTTQualityAnalyzer:
    """OTT/DCI 품질평가 도구 메인 클래스"""

    def __init__(self):
        self.metadata_analyzer = VideoMetadataAnalyzer()
        self.quality_calculator = QualityMetricsCalculator()
        self.standards_checker = DCIOTTStandardsChecker()
        self.html_generator = HTMLReportGenerator()

        self.version = "1.1.0"
        self.supported_formats = {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".3gp",
            ".ts",
            ".mts",
            ".mxf",
        }

    def print_banner(self):
        """프로그램 시작 배너 출력"""
        print("=" * 80)
        print("OTT/DCI 품질평가 도구 v{}".format(self.version))
        print("META(PQL) / NETFLIX(VMAF) 기준 종합 품질 분석")
        print("=" * 80)
        print()

    def print_menu(self):
        """메인 메뉴 출력"""
        print("사용 가능한 기능:")
        print("1. 비디오 메타데이터 분석")
        print("2. 품질 메트릭 비교 분석 (두 비디오)")
        print("3. 단일 비디오 품질 분석")
        print("4. DCI/OTT 표준 준수 검사")
        print("5. 종합 품질 분석 (모든 기능)")
        print("6. 배치 처리 (폴더 내 모든 비디오)")
        print("7. HTML 보고서 생성")
        print("0. 종료")
        print()

    def validate_file(self, file_path: str) -> bool:
        """파일 유효성 검사"""
        if not file_path or not file_path.strip():
            print("파일 경로를 입력해주세요.")
            return False

        file_path = file_path.strip().strip("\"'")

        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return False

        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            print(f"지원되지 않는 파일 형식입니다: {file_ext}")
            print(f"지원 형식: {', '.join(self.supported_formats)}")
            return False

        return True

    def get_file_input(self, prompt: str) -> Optional[str]:
        """파일 경로 입력 받기"""
        while True:
            file_path = input(prompt).strip().strip("\"'")

            if not file_path:
                return None

            if self.validate_file(file_path):
                return file_path

            retry = input("다시 시도하시겠습니까? (y/n): ").strip().lower()
            if retry not in ["y", "yes", "ㅇ"]:
                return None

    def save_results(
        self, results: Dict[str, Any], base_filename: str, suffix: str = ""
    ):
        """결과 저장"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if suffix:
                output_file = f"{base_filename}_{suffix}_{timestamp}.json"
            else:
                output_file = f"{base_filename}_{timestamp}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"결과가 저장되었습니다: {output_file}")
            return output_file
        except Exception as e:
            print(f"결과 저장 중 오류: {e}")
            return None

    def analyze_metadata(self):
        """1. 비디오 메타데이터 분석"""
        print("\n1. 비디오 메타데이터 분석")
        print("-" * 50)

        file_path = self.get_file_input("분석할 비디오 파일 경로: ")
        if not file_path:
            return

        try:
            print("메타데이터 분석 중...")
            result = self.metadata_analyzer.analyze_complete(file_path)

            # 결과 출력
            self.metadata_analyzer.print_summary(result)

            # 결과 저장
            base_name = Path(file_path).stem
            self.save_results(result, base_name, "metadata")

        except Exception as e:
            print(f"분석 중 오류 발생: {e}")

    def compare_quality_metrics(self):
        """2. 품질 메트릭 비교 분석"""
        print("\n2. 품질 메트릭 비교 분석")
        print("-" * 50)

        ref_file = self.get_file_input("참조 비디오 파일 경로: ")
        if not ref_file:
            return

        dist_file = self.get_file_input("비교할 비디오 파일 경로: ")
        if not dist_file:
            return

        # 옵션 설정
        try:
            frames_input = input("분석할 프레임 수 (기본값: 20): ").strip()
            num_frames = int(frames_input) if frames_input else 20
        except:
            num_frames = 20

        letterbox_input = input("레터박스 자동 제거 (y/n, 기본값: y): ").strip().lower()
        remove_letterbox = letterbox_input not in ["n", "no", "ㄴ"]

        try:
            print("품질 메트릭 계산 중...")
            result = self.quality_calculator.calculate_quality_metrics_comparison(
                ref_file, dist_file, num_frames, remove_letterbox
            )

            # 결과 출력
            self.quality_calculator.print_results(result)

            # 결과 저장
            base_name = f"{Path(ref_file).stem}_vs_{Path(dist_file).stem}"
            self.save_results(result, base_name, "quality_comparison")

        except Exception as e:
            print(f"품질 분석 중 오류 발생: {e}")

    def analyze_single_quality(self):
        """3. 단일 비디오 품질 분석"""
        print("\n3. 단일 비디오 품질 분석")
        print("-" * 50)

        file_path = self.get_file_input("분석할 비디오 파일 경로: ")
        if not file_path:
            return

        # 옵션 설정
        try:
            frames_input = input("분석할 프레임 수 (기본값: 20): ").strip()
            num_frames = int(frames_input) if frames_input else 20
        except:
            num_frames = 20

        try:
            print("품질 분석 중...")
            result = self.quality_calculator.calculate_single_video_metrics(
                file_path, num_frames
            )

            # 결과 출력
            self.quality_calculator.print_results(result)

            # 결과 저장
            base_name = Path(file_path).stem
            self.save_results(result, base_name, "quality_analysis")

        except Exception as e:
            print(f"품질 분석 중 오류 발생: {e}")

    def check_standards_compliance(self):
        """4. DCI/OTT 표준 준수 검사"""
        print("\n4. DCI/OTT 표준 준수 검사")
        print("-" * 50)

        file_path = self.get_file_input("검사할 비디오 파일 경로: ")
        if not file_path:
            return

        try:
            print("1단계: 메타데이터 분석 중...")
            metadata = self.metadata_analyzer.analyze_complete(file_path)

            print("2단계: DCI/OTT 표준 준수 검사 중...")
            metadata["analysis_timestamp"] = datetime.datetime.now().isoformat()
            result = self.standards_checker.run_comprehensive_check(file_path, metadata)

            # 결과 출력
            self.standards_checker.print_results(result)

            # 결과 저장
            base_name = Path(file_path).stem
            self.save_results(result, base_name, "standards_check")

        except Exception as e:
            print(f"표준 검사 중 오류 발생: {e}")

    def comprehensive_analysis(self):
        """5. 종합 품질 분석"""
        print("\n5. 종합 품질 분석 (모든 기능)")
        print("-" * 50)

        file_path = self.get_file_input("분석할 비디오 파일 경로: ")
        if not file_path:
            return

        # 비교 파일 옵션
        compare_option = (
            input("다른 비디오와 비교하시겠습니까? (y/n): ").strip().lower()
        )
        comparison_file = None

        if compare_option in ["y", "yes", "ㅇ"]:
            comparison_file = self.get_file_input("비교할 비디오 파일 경로: ")

        # 옵션 설정
        try:
            frames_input = input("분석할 프레임 수 (기본값: 20): ").strip()
            num_frames = int(frames_input) if frames_input else 20
        except:
            num_frames = 20

        try:
            comprehensive_results = {
                "analysis_info": {
                    "primary_file": file_path,
                    "comparison_file": comparison_file,
                    "analysis_timestamp": datetime.datetime.now().isoformat(),
                    "num_frames_analyzed": num_frames,
                }
            }

            # 1단계: 메타데이터 분석
            print("\n1단계: 메타데이터 분석 중...")
            metadata = self.metadata_analyzer.analyze_complete(file_path)
            comprehensive_results["metadata_analysis"] = metadata
            print("✓ 메타데이터 분석 완료")

            # 2단계: 품질 분석
            if comparison_file:
                print("\n2단계: 품질 메트릭 비교 분석 중...")
                quality_result = (
                    self.quality_calculator.calculate_quality_metrics_comparison(
                        file_path, comparison_file, num_frames, True
                    )
                )
                comprehensive_results["quality_comparison"] = quality_result
                print("✓ 품질 비교 분석 완료")
            else:
                print("\n2단계: 단일 비디오 품질 분석 중...")
                quality_result = self.quality_calculator.calculate_single_video_metrics(
                    file_path, num_frames
                )
                comprehensive_results["quality_analysis"] = quality_result
                print("✓ 품질 분석 완료")

            # 3단계: 표준 준수 검사
            print("\n3단계: DCI/OTT 표준 준수 검사 중...")
            metadata["analysis_timestamp"] = datetime.datetime.now().isoformat()
            standards_result = self.standards_checker.run_comprehensive_check(
                file_path, metadata
            )
            comprehensive_results["standards_compliance"] = standards_result
            print("✓ 표준 준수 검사 완료")

            # 결과 출력
            print("\n" + "=" * 80)
            print("종합 분석 결과")
            print("=" * 80)

            # 메타데이터 요약
            if "summary" in metadata:
                summary = metadata["summary"]
                print(f"\n파일 정보:")
                print(f"  파일명: {Path(file_path).name}")
                print(f"  해상도: {summary['resolution']}")
                print(f"  코덱: {summary['codec']}")
                print(f"  재생시간: {summary['duration_seconds']:.1f}초")
                print(f"  파일크기: {summary['file_size_mb']}MB")
                print(f"  비트레이트: {summary['bit_rate_kbps']}kbps")

            # 품질 분석 요약
            if comparison_file:
                print(f"\n품질 비교 분석:")
                if "psnr" in quality_result:
                    print(f"  PSNR: {quality_result['psnr']['mean_psnr']:.2f} dB")
                if "ssim" in quality_result:
                    print(f"  SSIM: {quality_result['ssim']['mean_ssim']:.4f}")
                if (
                    "vmaf" in quality_result
                    and quality_result["vmaf"].get("status") == "success"
                ):
                    print(f"  VMAF: {quality_result['vmaf']['mean_vmaf']:.2f}")
            else:
                print(f"\n품질 분석:")
                if "quality_analysis" in quality_result:
                    qa = quality_result["quality_analysis"]
                    print(f"  밝기: {qa['brightness']['mean']:.1f}")
                    print(f"  대비: {qa['contrast']['mean']:.1f}")
                    print(f"  선명도: {qa['sharpness']['mean']:.1f}")

            # 표준 준수 요약
            if "compliance_score" in standards_result:
                score = standards_result["compliance_score"]["overall_score"]
                print(f"\nDCI/OTT 표준 준수:")
                print(f"  종합 점수: {score['percentage']}% ({score['grade']})")

                summary = standards_result["compliance_score"]["summary"]
                print(
                    f"  검사 결과: 통과 {summary['passed']}, 경고 {summary['warnings']}, 실패 {summary['failures']}"
                )

            # 전체 결과 저장
            base_name = Path(file_path).stem
            output_file = self.save_results(
                comprehensive_results, base_name, "comprehensive"
            )

            # HTML 보고서 생성 옵션
            html_option = (
                input("\nHTML 보고서를 생성하시겠습니까? (y/n): ").strip().lower()
            )
            if html_option in ["y", "yes", "ㅇ"]:
                try:
                    html_file = self.html_generator.generate_comprehensive_report(
                        comprehensive_results, f"{base_name}_report.html"
                    )
                    print(f"HTML 보고서 생성 완료: {html_file}")

                    # 브라우저에서 열기 옵션
                    open_browser = (
                        input("브라우저에서 바로 열어보시겠습니까? (y/n): ")
                        .strip()
                        .lower()
                    )
                    if open_browser in ["y", "yes", "ㅇ"]:
                        import webbrowser

                        webbrowser.open(f"file://{os.path.abspath(html_file)}")

                except Exception as e:
                    print(f"HTML 보고서 생성 중 오류: {e}")

            print(f"\n종합 분석이 완료되었습니다!")
            if output_file:
                print(f"상세 결과는 {output_file}에서 확인할 수 있습니다.")

        except Exception as e:
            print(f"종합 분석 중 오류 발생: {e}")

    def batch_processing(self):
        """6. 배치 처리"""
        print("\n6. 배치 처리 (폴더 내 모든 비디오)")
        print("-" * 50)

        folder_path = input("비디오 파일이 있는 폴더 경로: ").strip().strip("\"'")

        if not folder_path or not os.path.exists(folder_path):
            print("폴더를 찾을 수 없습니다.")
            return

        # 비디오 파일 찾기
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(Path(folder_path).glob(f"*{ext}"))

        if not video_files:
            print(f"폴더에서 지원되는 비디오 파일을 찾을 수 없습니다.")
            print(f"지원 형식: {', '.join(self.supported_formats)}")
            return

        print(f"발견된 비디오 파일: {len(video_files)}개")
        for i, file in enumerate(video_files[:5], 1):
            print(f"  {i}. {file.name}")
        if len(video_files) > 5:
            print(f"  ... 및 {len(video_files) - 5}개 더")

        proceed = input("배치 처리를 시작하시겠습니까? (y/n): ").strip().lower()
        if proceed not in ["y", "yes", "ㅇ"]:
            return

        # 처리 옵션 선택
        print("\n처리 옵션:")
        print("1. 메타데이터 분석만")
        print("2. DCI/OTT 표준 검사")
        print("3. 품질 분석 (단일)")

        try:
            option = int(input("선택 (1-3): ").strip())
        except:
            option = 1

        batch_results = {
            "batch_info": {
                "folder_path": folder_path,
                "total_files": len(video_files),
                "processing_option": option,
                "start_time": datetime.datetime.now().isoformat(),
            },
            "results": [],
        }

        # 파일별 처리
        for i, video_file in enumerate(video_files, 1):
            print(f"\n처리 중 ({i}/{len(video_files)}): {video_file.name}")

            try:
                file_result = {
                    "file_name": video_file.name,
                    "file_path": str(video_file),
                    "processing_status": "success",
                }

                if option == 1:  # 메타데이터만
                    result = self.metadata_analyzer.analyze_complete(str(video_file))
                    file_result["metadata"] = result

                elif option == 2:  # 표준 검사
                    metadata = self.metadata_analyzer.analyze_complete(str(video_file))
                    metadata["analysis_timestamp"] = datetime.datetime.now().isoformat()
                    result = self.standards_checker.run_comprehensive_check(
                        str(video_file), metadata
                    )
                    file_result["standards_check"] = result

                elif option == 3:  # 품질 분석
                    result = self.quality_calculator.calculate_single_video_metrics(
                        str(video_file)
                    )
                    file_result["quality_analysis"] = result

                batch_results["results"].append(file_result)
                print(f"✓ 완료")

            except Exception as e:
                print(f"✗ 오류: {e}")
                file_result["processing_status"] = "error"
                file_result["error"] = str(e)
                batch_results["results"].append(file_result)

        batch_results["batch_info"]["end_time"] = datetime.datetime.now().isoformat()

        # 배치 결과 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_processing_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        # 요약 출력
        success_count = sum(
            1 for r in batch_results["results"] if r["processing_status"] == "success"
        )
        error_count = len(batch_results["results"]) - success_count

        print(f"\n배치 처리 완료!")
        print(f"성공: {success_count}개, 실패: {error_count}개")
        print(f"결과 저장: {output_file}")

    def generate_html_report(self):
        """7. HTML 보고서 생성"""
        print("\n7. HTML 보고서 생성")
        print("-" * 50)

        # JSON 파일 선택
        json_file = input("분석 결과 JSON 파일 경로: ").strip().strip("\"'")

        if not json_file or not os.path.exists(json_file):
            print("JSON 파일을 찾을 수 없습니다.")
            return

        try:
            # JSON 파일 읽기
            with open(json_file, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)

            # HTML 보고서 생성
            print("HTML 보고서 생성 중...")
            base_name = Path(json_file).stem
            html_file = self.html_generator.generate_comprehensive_report(
                analysis_data, f"{base_name}_report.html"
            )

            print(f"HTML 보고서 생성 완료: {html_file}")

            # 브라우저에서 열기 옵션
            open_browser = (
                input("브라우저에서 바로 열어보시겠습니까? (y/n): ").strip().lower()
            )
            if open_browser in ["y", "yes", "ㅇ"]:
                import webbrowser

                webbrowser.open(f"file://{os.path.abspath(html_file)}")
                print("브라우저에서 보고서를 열었습니다.")

        except Exception as e:
            print(f"HTML 보고서 생성 중 오류: {e}")

    def run(self):
        """메인 실행 함수"""
        self.print_banner()

        while True:
            self.print_menu()

            try:
                choice = input("기능을 선택하세요 (0-7): ").strip()

                if choice == "0":
                    print("프로그램을 종료합니다.")
                    break
                elif choice == "1":
                    self.analyze_metadata()
                elif choice == "2":
                    self.compare_quality_metrics()
                elif choice == "3":
                    self.analyze_single_quality()
                elif choice == "4":
                    self.check_standards_compliance()
                elif choice == "5":
                    self.comprehensive_analysis()
                elif choice == "6":
                    self.batch_processing()
                elif choice == "7":
                    self.generate_html_report()
                else:
                    print("잘못된 선택입니다. 0-7 사이의 숫자를 입력하세요.")

                input("\n계속하려면 Enter를 누르세요...")
                print("\n" + "=" * 80 + "\n")

            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n예상치 못한 오류가 발생했습니다: {e}")
                input("계속하려면 Enter를 누르세요...")


def main():
    """메인 함수"""
    try:
        analyzer = OTTQualityAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"프로그램 시작 중 오류: {e}")


if __name__ == "__main__":
    main()
