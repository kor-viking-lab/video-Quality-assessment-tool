import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class HTMLReportGenerator:
    """
    OTT/DCI 품질평가 결과를 HTML 보고서로 생성하는 클래스
    """

    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.output_dir = Path("reports_html")
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(
        self, analysis_data: Dict[str, Any], output_filename: Optional[str] = None
    ) -> str:
        """종합 분석 결과 HTML 보고서 생성"""
        try:
            # 출력 파일명 생성
            if not output_filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"quality_report_{timestamp}.html"

            output_path = self.output_dir / output_filename

            # HTML 생성
            html_content = self._create_html_template(analysis_data)

            # 파일 저장
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"HTML 보고서가 생성되었습니다: {output_path}")
            return str(output_path)

        except Exception as e:
            raise Exception(f"HTML 보고서 생성 중 오류: {str(e)}")

    def _create_html_template(self, data: Dict[str, Any]) -> str:
        """HTML 템플릿 생성"""

        # 데이터 추출
        analysis_info = data.get("analysis_info", {})
        metadata = data.get("metadata_analysis", {})
        # 업스케일링, 디노이즈, 색복원 모두 지원
        quality_analysis = (
            data.get("quality_analysis")
            or data.get("quality_comparison")
            or data.get("denoise_evaluation")
            or data.get("colorization_evaluation")
        )
        standards_compliance = data.get("standards_compliance", {})

        # HTML 생성
        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OTT/DCI 품질평가 보고서</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script>
        // Chart.js 로드 실패 시 대체 메시지
        window.addEventListener('load', function() {{
            if (typeof Chart === 'undefined') {{
                console.error('Chart.js 로드 실패');
                document.body.insertAdjacentHTML('afterbegin', 
                    '<div style="background: #f8d7da; color: #721c24; padding: 10px; text-align: center; border: 1px solid #f5c6cb;">' +
                    '⚠️ 차트 라이브러리를 로드할 수 없습니다. 인터넷 연결을 확인하세요.' +
                    '</div>'
                );
            }}
        }});
    </script>
</head>
<body>
    <div class="container">
        {self._generate_header(analysis_info)}
        {self._generate_summary_section(metadata, quality_analysis, standards_compliance)}
        {self._generate_metadata_section(metadata)}
        {self._generate_quality_section(quality_analysis)}
        {self._generate_standards_section(standards_compliance, quality_analysis)}
        {self._generate_charts_section(quality_analysis)}
        {self._generate_footer()}
    </div>
    
    <script>
        // 품질 데이터 주입
        {self._inject_chart_data(quality_analysis)}
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html

    def _get_css_styles(self) -> str:
        """CSS 스타일 정의"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header .subtitle {
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        
        .section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .section h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            transform: translateY(0);
            transition: transform 0.3s ease;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
        }
        
        .summary-card h3 {
            font-size: 1.1em;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .summary-card .unit {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }
        
        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .info-item .label {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .info-item .value {
            color: #34495e;
            font-weight: 500;
        }
        
        .chart-container {
            position: relative;
            height: 400px;
            margin: 30px 0;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            color: white;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _generate_header(self, analysis_info: Dict[str, Any]) -> str:
        """헤더 섹션 생성"""
        primary_file = analysis_info.get("primary_file", "알 수 없음")
        timestamp = analysis_info.get("analysis_timestamp", "")

        if timestamp:
            try:
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y년 %m월 %d일 %H:%M:%S")
            except:
                formatted_time = timestamp
        else:
            formatted_time = "알 수 없음"

        return f"""
        <div class="header">
            <h1>📊 OTT/DCI 품질평가 보고서</h1>
            <div class="subtitle">META(PQL) / NETFLIX(VMAF) 기준 종합 분석</div>
            <div>
                <strong>분석 파일:</strong> {Path(primary_file).name}<br>
                <strong>분석 시간:</strong> {formatted_time}
            </div>
        </div>
        """

    def _generate_summary_section(
        self, metadata: Dict, quality: Dict, standards: Dict
    ) -> str:
        """요약 섹션 생성"""
        summary_cards = []

        # 파일 정보
        if metadata and "summary" in metadata:
            summary = metadata["summary"]
            summary_cards.append(
                f"""
                <div class="summary-card">
                    <h3>해상도</h3>
                    <div class="value">{summary.get('resolution', 'N/A')}</div>
                </div>
            """
            )

            summary_cards.append(
                f"""
                <div class="summary-card">
                    <h3>코덱</h3>
                    <div class="value">{summary.get('codec', 'N/A')}</div>
                </div>
            """
            )

            summary_cards.append(
                f"""
                <div class="summary-card">
                    <h3>파일 크기</h3>
                    <div class="value">{summary.get('file_size_mb', 0)}</div>
                    <div class="unit">MB</div>
                </div>
            """
            )

        # 품질 점수
        if quality:
            if "psnr" in quality:
                psnr_val = quality["psnr"].get("mean_psnr", 0)
                summary_cards.append(
                    f"""
                    <div class="summary-card">
                        <h3>PSNR</h3>
                        <div class="value">{psnr_val:.1f}</div>
                        <div class="unit">dB</div>
                    </div>
                """
                )

            if "ssim" in quality:
                ssim_val = quality["ssim"].get("mean_ssim", 0)
                summary_cards.append(
                    f"""
                    <div class="summary-card">
                        <h3>SSIM</h3>
                        <div class="value">{ssim_val:.3f}</div>
                    </div>
                """
                )

        return f"""
        <div class="section">
            <h2>📋 분석 요약</h2>
            <div class="summary-grid">
                {''.join(summary_cards)}
            </div>
        </div>
        """

    def _generate_metadata_section(self, metadata: Dict) -> str:
        """메타데이터 섹션 생성"""
        if not metadata:
            return ""

        info_items = []

        # FFmpeg 데이터
        if "ffmpeg_data" in metadata:
            ffmpeg_data = metadata["ffmpeg_data"]
            if "video" in ffmpeg_data:
                video = ffmpeg_data["video"]
                info_items.extend(
                    [
                        ("코덱 이름", video.get("codec_name", "N/A")),
                        ("해상도", f"{video.get('width', 0)}x{video.get('height', 0)}"),
                        ("프레임레이트", f"{video.get('frame_rate', 0)} fps"),
                        ("픽셀 포맷", video.get("pixel_format", "N/A")),
                        (
                            "비트레이트",
                            f"{video.get('bit_rate', 0) // 1000 if video.get('bit_rate') else 0} kbps",
                        ),
                    ]
                )

        info_html = ""
        for label, value in info_items:
            info_html += f"""
                <div class="info-item">
                    <span class="label">{label}</span>
                    <span class="value">{value}</span>
                </div>
            """

        return f"""
        <div class="section">
            <h2>📁 파일 메타데이터</h2>
            <div class="info-grid">
                {info_html}
            </div>
        </div>
        """

    def _generate_quality_section(self, quality: Dict) -> str:
        """품질 분석 섹션 생성"""
        if not quality:
            return ""

        quality_html = ""
        evaluation_type = quality.get("evaluation_type", "")

        # 공통 메트릭: PSNR, SSIM
        if "psnr" in quality:
            psnr_data = quality["psnr"]
            quality_html += f"""
                <div class="info-item">
                    <span class="label">PSNR 평균</span>
                    <span class="value">{psnr_data.get('mean_psnr', 0):.2f} dB</span>
                </div>
            """

        if "ssim" in quality:
            ssim_data = quality["ssim"]
            quality_html += f"""
                <div class="info-item">
                    <span class="label">SSIM 평균</span>
                    <span class="value">{ssim_data.get('mean_ssim', 0):.4f}</span>
                </div>
            """

        # VMAF (업스케일링)
        if "vmaf" in quality and quality["vmaf"].get("status") == "success":
            vmaf_data = quality["vmaf"]
            quality_html += f"""
                <div class="info-item">
                    <span class="label">VMAF 평균</span>
                    <span class="value">{vmaf_data.get('mean_vmaf', 0):.2f}</span>
                </div>
            """

        # 디노이즈 특화 메트릭
        if evaluation_type == "denoising":
            if "noise_reduction" in quality:
                nr_data = quality["noise_reduction"]
                quality_html += f"""
                    <div class="info-item">
                        <span class="label">노이즈 제거율</span>
                        <span class="value">{nr_data.get('mean_reduction_ratio', 0):.1f}%</span>
                    </div>
                    <div class="info-item">
                        <span class="label">원본 노이즈 레벨</span>
                        <span class="value">{nr_data.get('mean_original_noise', 0):.2f}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">디노이즈 후 노이즈 레벨</span>
                        <span class="value">{nr_data.get('mean_denoised_noise', 0):.2f}</span>
                    </div>
                """

            if "detail_preservation" in quality:
                dp_data = quality["detail_preservation"]
                quality_html += f"""
                    <div class="info-item">
                        <span class="label">디테일 보존율</span>
                        <span class="value">{dp_data.get('mean_preservation_ratio', 0):.1f}%</span>
                    </div>
                """

            if "blur_amount" in quality:
                blur_data = quality["blur_amount"]
                quality_html += f"""
                    <div class="info-item">
                        <span class="label">선명도 점수</span>
                        <span class="value">{blur_data.get('mean_blur_score', 0):.1f}</span>
                    </div>
                """

        # 색복원 특화 메트릭
        elif evaluation_type == "colorization":
            if "color_accuracy" in quality:
                ca_data = quality["color_accuracy"]
                quality_html += f"""
                    <div class="info-item">
                        <span class="label">색상 정확도 (Delta E)</span>
                        <span class="value">{ca_data.get('mean_delta_e', 0):.2f}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">밝기 차이 (L)</span>
                        <span class="value">{ca_data.get('mean_l_diff', 0):.2f}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">색상 채널 차이 (a)</span>
                        <span class="value">{ca_data.get('mean_a_diff', 0):.2f}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">색상 채널 차이 (b)</span>
                        <span class="value">{ca_data.get('mean_b_diff', 0):.2f}</span>
                    </div>
                """

            if "saturation_metrics" in quality:
                sat_data = quality["saturation_metrics"]
                quality_html += f"""
                    <div class="info-item">
                        <span class="label">채도 복원율</span>
                        <span class="value">{sat_data.get('mean_saturation_ratio', 0):.1f}%</span>
                    </div>
                """

            if "hue_difference" in quality:
                hue_data = quality["hue_difference"]
                quality_html += f"""
                    <div class="info-item">
                        <span class="label">색조 차이</span>
                        <span class="value">{hue_data.get('mean_hue_diff', 0):.2f}°</span>
                    </div>
                """

        return f"""
        <div class="section">
            <h2>🎯 품질 분석 결과</h2>
            <div class="info-grid">
                {quality_html}
            </div>
        </div>
        """

    def _generate_standards_section(self, standards: Dict, quality: Dict = None) -> str:
        """표준 준수 섹션 생성"""
        if not standards or "compliance_score" not in standards:
            return ""

        # 디노이즈와 색복원에서는 DCI/OTT 표준 준수도 섹션을 숨김
        if quality:
            evaluation_type = quality.get("evaluation_type", "")
            if evaluation_type in ["denoising", "colorization"]:
                return ""

        score = standards["compliance_score"]["overall_score"]

        return f"""
        <div class="section">
            <h2>📊 DCI/OTT 표준 준수도</h2>
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="display: inline-block; padding: 10px 20px; border-radius: 25px; background: #27ae60; color: white; font-weight: bold; font-size: 1.2em;">
                    {score['grade']} ({score['percentage']:.1f}%)
                </div>
            </div>
        </div>
        """

    def _generate_charts_section(self, quality: Dict) -> str:
        """차트 섹션 생성"""
        if not quality:
            return ""

        charts_html = ""

        # PSNR 차트
        if "psnr" in quality and "frame_psnr_values" in quality["psnr"]:
            charts_html += """
                <div class="chart-container">
                    <canvas id="psnrChart"></canvas>
                </div>
            """

        # SSIM 차트
        if "ssim" in quality and "frame_ssim_values" in quality["ssim"]:
            charts_html += """
                <div class="chart-container">
                    <canvas id="ssimChart"></canvas>
                </div>
            """

        if charts_html:
            return f"""
            <div class="section">
                <h2>📈 품질 지표 차트</h2>
                {charts_html}
            </div>
            """

        return ""

    def _generate_footer(self) -> str:
        """푸터 생성"""
        return f"""
        <div class="footer">
            <p>OTT/DCI 품질평가 도구 v1.1.0 | 생성 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>META(PQL) / NETFLIX(VMAF) 기준 종합 품질 분석</p>
        </div>
        """

    def _inject_chart_data(self, quality: Dict) -> str:
        """차트 데이터를 JavaScript에 주입"""
        data_injection = ""

        if quality:
            print(f"차트 데이터 주입 중... quality keys: {list(quality.keys())}")

            # PSNR 데이터
            if "psnr" in quality and "frame_psnr_values" in quality["psnr"]:
                psnr_values = quality["psnr"]["frame_psnr_values"]
                print(f"PSNR 데이터 발견: {len(psnr_values)}개 프레임")
                data_injection += f"const psnrData = {json.dumps(psnr_values)};\n"
                data_injection += f"console.log('PSNR 데이터 로드됨:', psnrData);\n"
            else:
                print("PSNR 프레임 데이터 없음 - 테스트 데이터 생성")
                test_psnr = [30.5, 31.2, 29.8, 32.1, 30.9, 31.8, 29.5, 33.2, 31.1, 30.7]
                data_injection += f"const psnrData = {json.dumps(test_psnr)};\n"
                data_injection += (
                    f"console.log('PSNR 테스트 데이터 생성됨:', psnrData);\n"
                )

            # SSIM 데이터
            if "ssim" in quality and "frame_ssim_values" in quality["ssim"]:
                ssim_values = quality["ssim"]["frame_ssim_values"]
                print(f"SSIM 데이터 발견: {len(ssim_values)}개 프레임")
                data_injection += f"const ssimData = {json.dumps(ssim_values)};\n"
                data_injection += f"console.log('SSIM 데이터 로드됨:', ssimData);\n"
            else:
                print("SSIM 프레임 데이터 없음 - 테스트 데이터 생성")
                test_ssim = [0.85, 0.87, 0.83, 0.89, 0.86, 0.88, 0.82, 0.91, 0.87, 0.85]
                data_injection += f"const ssimData = {json.dumps(test_ssim)};\n"
                data_injection += (
                    f"console.log('SSIM 테스트 데이터 생성됨:', ssimData);\n"
                )
        else:
            print("품질 데이터가 전혀 없음 - 테스트 데이터 생성")
            test_psnr = [30.5, 31.2, 29.8, 32.1, 30.9, 31.8, 29.5, 33.2, 31.1, 30.7]
            test_ssim = [0.85, 0.87, 0.83, 0.89, 0.86, 0.88, 0.82, 0.91, 0.87, 0.85]
            data_injection += f"const psnrData = {json.dumps(test_psnr)};\n"
            data_injection += f"const ssimData = {json.dumps(test_ssim)};\n"
            data_injection += "console.log('모든 테스트 데이터 생성됨');\n"

        return data_injection

    def _get_javascript(self) -> str:
        """JavaScript 코드 생성"""
        return """
        // 차트 생성 함수
        function createChart(canvasId, data, label, color) {
            const ctx = document.getElementById(canvasId);
            if (!ctx) {
                console.error(`Canvas 요소를 찾을 수 없습니다: ${canvasId}`);
                return;
            }
            
            console.log(`${label} 차트 생성 중... 데이터 길이: ${data.length}`);
            
            try {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.map((_, i) => `Frame ${i + 1}`),
                        datasets: [{
                            label: label,
                            data: data,
                            borderColor: color,
                            backgroundColor: color + '20',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: label + ' 프레임별 변화',
                                font: { size: 16 }
                            },
                            legend: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                grid: {
                                    color: '#e0e0e0'
                                }
                            },
                            x: {
                                grid: {
                                    color: '#e0e0e0'
                                }
                            }
                        }
                    }
                });
                console.log(`${label} 차트 생성 완료`);
            } catch (error) {
                console.error(`${label} 차트 생성 중 오류:`, error);
            }
        }
        
        // 페이지 로드 후 차트 생성
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM 로드 완료, 차트 생성 시작');
            
            // Chart.js 로드 확인
            if (typeof Chart === 'undefined') {
                console.error('Chart.js가 로드되지 않았습니다!');
                return;
            }
            
            // 데이터 존재 확인 및 차트 생성
            if (typeof psnrData !== 'undefined' && psnrData.length > 0) {
                console.log('PSNR 차트 생성 중...', psnrData);
                createChart('psnrChart', psnrData, 'PSNR (dB)', '#667eea');
            }
            
            if (typeof ssimData !== 'undefined' && ssimData.length > 0) {
                console.log('SSIM 차트 생성 중...', ssimData);
                createChart('ssimChart', ssimData, 'SSIM', '#764ba2');
            }
        });
        """


def main():
    """테스트용 메인 함수"""
    import datetime

    # 샘플 데이터 생성
    sample_data = {
        "analysis_info": {
            "primary_file": "test_video.mp4",
            "analysis_timestamp": datetime.datetime.now().isoformat(),
        },
        "metadata_analysis": {
            "summary": {
                "resolution": "3840x2160",
                "codec": "HEVC",
                "file_size_mb": 150.5,
                "duration_seconds": 120.0,
            }
        },
        "quality_analysis": {
            "psnr": {
                "mean_psnr": 32.5,
                "min_psnr": 28.1,
                "max_psnr": 35.2,
                "frame_psnr_values": [30.1, 31.5, 32.8, 33.2, 32.1],
            },
            "ssim": {
                "mean_ssim": 0.85,
                "min_ssim": 0.82,
                "max_ssim": 0.88,
                "frame_ssim_values": [0.83, 0.85, 0.87, 0.86, 0.84],
            },
        },
        "standards_compliance": {
            "compliance_score": {
                "overall_score": {"percentage": 85.5, "grade": "A (양호)"},
                "recommendations": [
                    "Rec. 2020 색상 공간 사용 권장",
                    "10-bit 이상의 비트 깊이 사용",
                ],
            }
        },
    }

    generator = HTMLReportGenerator()
    report_path = generator.generate_comprehensive_report(sample_data)
    print(f"샘플 보고서 생성됨: {report_path}")


if __name__ == "__main__":
    main()
