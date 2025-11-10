import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import ffmpeg
from pymediainfo import MediaInfo
import re


class DCIOTTStandardsChecker:
    """
    DCI/OTT 표준 준수 검사기
    META(PQL) / NETFLIX(VMAF) 품질평가 기준에 따른 종합 검사
    """

    def __init__(self):
        # 표준 규격 정의
        self.standards = {
            "containers": {
                "required": ["mp4", "mov", "imf", "mxf"],
                "supported": ["jpeg2000", "mkv"],
            },
            "codecs": {
                "premium": ["hevc", "h265", "xavc-intra", "prores"],
                "acceptable": ["h264", "avc"],
            },
            "bitrates": {
                "streaming": {"min": 10, "max": 20, "unit": "Mbps"},
                "production": {"min": 320, "max": 480, "unit": "Mbps"},
            },
            "color_systems": {
                "required": ["rec2020", "bt2020"],
                "advanced": ["rec2100", "bt2100"],
                "legacy": ["rec709", "bt709"],
            },
            "bit_depth": {"minimum": 10, "recommended": 12},
            "resolutions": {
                "uhd": {"width": 3840, "height": 2160, "name": "UHD 4K"},
                "dci_4k": {"width": 4096, "height": 2160, "name": "DCI 4K"},
                "aspect_ratios": ["1.85:1", "2.39:1", "16:9"],
            },
            "frame_rates": {
                "standard": [24, 25, 30, 60],
                "cinema": [24],
                "broadcast": [25, 30, 50, 60],
            },
            "hdr_formats": {
                "supported": ["hdr10", "dolby vision", "hlg", "hybrid log-gamma"],
                "metadata_required": ["maxfall", "maxcll", "mastering display"],
            },
            "luminance": {
                "white_point": {"min": 1000, "recommended": 4000, "unit": "nits"},
                "black_level": {"max": 0.005, "recommended": 0.0005, "unit": "nits"},
            },
        }

    def check_container_format(self, file_path: str, metadata: Dict) -> Dict[str, Any]:
        """컨테이너 형식 검사"""
        try:
            file_ext = Path(file_path).suffix.lower().lstrip(".")
            format_name = (
                metadata.get("mediainfo_data", {})
                .get("file_info", {})
                .get("format", "")
                .lower()
            )

            result = {
                "file_extension": file_ext,
                "detected_format": format_name,
                "compliance": {},
            }

            # 확장자 검사
            if file_ext in self.standards["containers"]["required"]:
                result["compliance"]["extension"] = {
                    "status": "pass",
                    "level": "required",
                }
            elif file_ext in self.standards["containers"]["supported"]:
                result["compliance"]["extension"] = {
                    "status": "pass",
                    "level": "supported",
                }
            else:
                result["compliance"]["extension"] = {
                    "status": "fail",
                    "reason": f"지원되지 않는 확장자: {file_ext}",
                }

            # 포맷 검사
            format_keywords = ["mp4", "mov", "quicktime", "isom", "mxf", "imf"]
            format_detected = any(keyword in format_name for keyword in format_keywords)

            if format_detected:
                result["compliance"]["format"] = {"status": "pass"}
            else:
                result["compliance"]["format"] = {
                    "status": "warning",
                    "reason": f"알 수 없는 포맷: {format_name}",
                }

            return result

        except Exception as e:
            return {"error": f"컨테이너 형식 검사 중 오류: {str(e)}"}

    def check_video_codec(self, metadata: Dict) -> Dict[str, Any]:
        """비디오 코덱 검사"""
        try:
            ffmpeg_codec = (
                metadata.get("ffmpeg_data", {})
                .get("video", {})
                .get("codec_name", "")
                .lower()
            )
            mediainfo_codec = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("codec_name", "")
                .lower()
            )

            result = {
                "detected_codecs": {
                    "ffmpeg": ffmpeg_codec,
                    "mediainfo": mediainfo_codec,
                },
                "compliance": {},
            }

            # 코덱 정규화 (다양한 표기법 통합)
            codec_variants = {
                "hevc": ["hevc", "h265", "h.265"],
                "h264": ["h264", "h.264", "avc", "avc1"],
                "prores": ["prores", "prores_ks", "apple prores"],
                "xavc": ["xavc", "xavc-intra"],
            }

            detected_standard_codec = None
            for standard_codec, variants in codec_variants.items():
                if any(
                    variant in ffmpeg_codec or variant in mediainfo_codec
                    for variant in variants
                ):
                    detected_standard_codec = standard_codec
                    break

            if detected_standard_codec:
                if detected_standard_codec in ["hevc", "prores", "xavc"]:
                    result["compliance"]["codec"] = {
                        "status": "pass",
                        "level": "premium",
                        "codec": detected_standard_codec,
                    }
                elif detected_standard_codec in ["h264"]:
                    result["compliance"]["codec"] = {
                        "status": "pass",
                        "level": "acceptable",
                        "codec": detected_standard_codec,
                        "note": "H.264는 허용되지만 HEVC 권장",
                    }
            else:
                result["compliance"]["codec"] = {
                    "status": "fail",
                    "reason": f"지원되지 않는 코덱: {ffmpeg_codec}",
                }

            return result

        except Exception as e:
            return {"error": f"비디오 코덱 검사 중 오류: {str(e)}"}

    def check_bitrate(self, metadata: Dict) -> Dict[str, Any]:
        """비트레이트 검사"""
        try:
            # 비트레이트 정보 수집
            video_bitrate = (
                metadata.get("ffmpeg_data", {}).get("video", {}).get("bit_rate")
            )
            overall_bitrate = (
                metadata.get("mediainfo_data", {})
                .get("file_info", {})
                .get("overall_bit_rate")
            )

            # bps를 Mbps로 변환
            video_bitrate_mbps = (video_bitrate / 1_000_000) if video_bitrate else None
            overall_bitrate_mbps = (
                (overall_bitrate / 1_000_000) if overall_bitrate else None
            )

            result = {
                "detected_bitrates": {
                    "video_mbps": (
                        round(video_bitrate_mbps, 2) if video_bitrate_mbps else None
                    ),
                    "overall_mbps": (
                        round(overall_bitrate_mbps, 2) if overall_bitrate_mbps else None
                    ),
                },
                "compliance": {},
            }

            # 비트레이트 기준으로 용도 판단
            primary_bitrate = video_bitrate_mbps or overall_bitrate_mbps

            if primary_bitrate:
                streaming_range = self.standards["bitrates"]["streaming"]
                production_range = self.standards["bitrates"]["production"]

                if streaming_range["min"] <= primary_bitrate <= streaming_range["max"]:
                    result["compliance"]["bitrate"] = {
                        "status": "pass",
                        "category": "streaming",
                        "range": f"{streaming_range['min']}-{streaming_range['max']} Mbps",
                    }
                elif (
                    production_range["min"]
                    <= primary_bitrate
                    <= production_range["max"]
                ):
                    result["compliance"]["bitrate"] = {
                        "status": "pass",
                        "category": "production",
                        "range": f"{production_range['min']}-{production_range['max']} Mbps",
                    }
                elif primary_bitrate < streaming_range["min"]:
                    result["compliance"]["bitrate"] = {
                        "status": "warning",
                        "reason": f'비트레이트가 낮음: {primary_bitrate:.2f} Mbps (최소 {streaming_range["min"]} Mbps 권장)',
                    }
                else:
                    result["compliance"]["bitrate"] = {
                        "status": "pass",
                        "category": "high_quality",
                        "note": f"고품질 비트레이트: {primary_bitrate:.2f} Mbps",
                    }
            else:
                result["compliance"]["bitrate"] = {
                    "status": "fail",
                    "reason": "비트레이트 정보를 찾을 수 없음",
                }

            return result

        except Exception as e:
            return {"error": f"비트레이트 검사 중 오류: {str(e)}"}

    def check_color_system(self, metadata: Dict) -> Dict[str, Any]:
        """색상 시스템 검사"""
        try:
            color_space = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("color_space", "")
                .lower()
            )
            color_primaries = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("color_primaries", "")
                .lower()
            )
            transfer_characteristics = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("transfer_characteristics", "")
                .lower()
            )
            matrix_coefficients = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("matrix_coefficients", "")
                .lower()
            )

            result = {
                "detected_color_info": {
                    "color_space": color_space,
                    "color_primaries": color_primaries,
                    "transfer_characteristics": transfer_characteristics,
                    "matrix_coefficients": matrix_coefficients,
                },
                "compliance": {},
            }

            # 색상 시스템 분류
            if any(
                keyword in color_primaries
                for keyword in ["bt2020", "rec2020", "rec. 2020"]
            ):
                result["compliance"]["color_system"] = {
                    "status": "pass",
                    "level": "premium",
                    "standard": "Rec. 2020 (Wide Color Gamut)",
                }
            elif any(keyword in color_primaries for keyword in ["bt2100", "rec2100"]):
                result["compliance"]["color_system"] = {
                    "status": "pass",
                    "level": "advanced",
                    "standard": "Rec. 2100 HLG",
                }
            elif any(keyword in color_primaries for keyword in ["bt709", "rec709"]):
                result["compliance"]["color_system"] = {
                    "status": "pass",
                    "level": "legacy",
                    "standard": "Rec. 709",
                    "note": "Rec. 2020 권장",
                }
            else:
                result["compliance"]["color_system"] = {
                    "status": "warning",
                    "reason": f"알 수 없는 색상 시스템: {color_primaries}",
                }

            return result

        except Exception as e:
            return {"error": f"색상 시스템 검사 중 오류: {str(e)}"}

    def check_bit_depth(self, metadata: Dict) -> Dict[str, Any]:
        """비트 깊이 검사"""
        try:
            bit_depth = (
                metadata.get("mediainfo_data", {}).get("video", {}).get("bit_depth")
            )

            result = {"detected_bit_depth": bit_depth, "compliance": {}}

            if bit_depth:
                bit_depth = int(bit_depth)

                if bit_depth >= self.standards["bit_depth"]["recommended"]:
                    result["compliance"]["bit_depth"] = {
                        "status": "pass",
                        "level": "recommended",
                        "value": bit_depth,
                    }
                elif bit_depth >= self.standards["bit_depth"]["minimum"]:
                    result["compliance"]["bit_depth"] = {
                        "status": "pass",
                        "level": "minimum",
                        "value": bit_depth,
                        "note": f"{self.standards['bit_depth']['recommended']}-bit 권장",
                    }
                else:
                    result["compliance"]["bit_depth"] = {
                        "status": "fail",
                        "reason": f'비트 깊이 부족: {bit_depth}-bit (최소 {self.standards["bit_depth"]["minimum"]}-bit 필요)',
                    }
            else:
                result["compliance"]["bit_depth"] = {
                    "status": "warning",
                    "reason": "비트 깊이 정보를 찾을 수 없음",
                }

            return result

        except Exception as e:
            return {"error": f"비트 깊이 검사 중 오류: {str(e)}"}

    def check_resolution_and_aspect_ratio(self, metadata: Dict) -> Dict[str, Any]:
        """해상도 및 화면비 검사"""
        try:
            width = metadata.get("ffmpeg_data", {}).get("video", {}).get("width")
            height = metadata.get("ffmpeg_data", {}).get("video", {}).get("height")
            aspect_ratio = (
                metadata.get("ffmpeg_data", {}).get("video", {}).get("aspect_ratio")
            )

            result = {
                "detected_resolution": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                },
                "compliance": {},
            }

            if width and height:
                # 해상도 검사
                if width == 3840 and height == 2160:
                    result["compliance"]["resolution"] = {
                        "status": "pass",
                        "standard": "UHD 4K",
                        "resolution": f"{width}x{height}",
                    }
                elif width == 4096 and height == 2160:
                    result["compliance"]["resolution"] = {
                        "status": "pass",
                        "standard": "DCI 4K",
                        "resolution": f"{width}x{height}",
                    }
                elif width >= 1920 and height >= 1080:
                    result["compliance"]["resolution"] = {
                        "status": "pass",
                        "standard": "HD/FHD",
                        "resolution": f"{width}x{height}",
                        "note": "4K 해상도 권장",
                    }
                else:
                    result["compliance"]["resolution"] = {
                        "status": "fail",
                        "reason": f"해상도 부족: {width}x{height} (최소 1920x1080 필요)",
                    }

                # 화면비 검사
                calculated_ratio = round(width / height, 2)
                aspect_ratios_decimal = {
                    "16:9": round(16 / 9, 2),
                    "1.85:1": 1.85,
                    "2.39:1": 2.39,
                }

                matching_ratio = None
                for ratio_name, ratio_value in aspect_ratios_decimal.items():
                    if abs(calculated_ratio - ratio_value) < 0.05:  # 허용 오차
                        matching_ratio = ratio_name
                        break

                if matching_ratio:
                    result["compliance"]["aspect_ratio"] = {
                        "status": "pass",
                        "ratio": matching_ratio,
                        "calculated": f"{calculated_ratio:.2f}:1",
                    }
                else:
                    result["compliance"]["aspect_ratio"] = {
                        "status": "warning",
                        "reason": f"비표준 화면비: {calculated_ratio:.2f}:1",
                        "supported": list(aspect_ratios_decimal.keys()),
                    }
            else:
                result["compliance"]["resolution"] = {
                    "status": "fail",
                    "reason": "해상도 정보를 찾을 수 없음",
                }

            return result

        except Exception as e:
            return {"error": f"해상도 검사 중 오류: {str(e)}"}

    def check_frame_rate(self, metadata: Dict) -> Dict[str, Any]:
        """프레임 레이트 검사"""
        try:
            frame_rate = (
                metadata.get("ffmpeg_data", {}).get("video", {}).get("frame_rate")
            )

            result = {"detected_frame_rate": frame_rate, "compliance": {}}

            if frame_rate:
                frame_rate = float(frame_rate)

                if frame_rate in self.standards["frame_rates"]["standard"]:
                    if frame_rate == 24:
                        result["compliance"]["frame_rate"] = {
                            "status": "pass",
                            "category": "cinema",
                            "value": frame_rate,
                        }
                    elif frame_rate in [25, 30, 50, 60]:
                        result["compliance"]["frame_rate"] = {
                            "status": "pass",
                            "category": "broadcast",
                            "value": frame_rate,
                        }
                else:
                    # 근사치 검사 (23.976fps 등)
                    tolerance = 0.1
                    for standard_fps in self.standards["frame_rates"]["standard"]:
                        if abs(frame_rate - standard_fps) < tolerance:
                            result["compliance"]["frame_rate"] = {
                                "status": "pass",
                                "category": "standard_variant",
                                "value": frame_rate,
                                "note": f"표준 {standard_fps}fps의 변형",
                            }
                            break
                    else:
                        result["compliance"]["frame_rate"] = {
                            "status": "warning",
                            "reason": f"비표준 프레임레이트: {frame_rate}fps",
                            "supported": self.standards["frame_rates"]["standard"],
                        }
            else:
                result["compliance"]["frame_rate"] = {
                    "status": "warning",
                    "reason": "프레임레이트 정보를 찾을 수 없음",
                }

            return result

        except Exception as e:
            return {"error": f"프레임레이트 검사 중 오류: {str(e)}"}

    def check_hdr_support(self, metadata: Dict) -> Dict[str, Any]:
        """HDR 지원 검사"""
        try:
            transfer_characteristics = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("transfer_characteristics", "")
                .lower()
            )
            color_primaries = (
                metadata.get("mediainfo_data", {})
                .get("video", {})
                .get("color_primaries", "")
                .lower()
            )

            result = {
                "detected_hdr_info": {
                    "transfer_characteristics": transfer_characteristics,
                    "color_primaries": color_primaries,
                },
                "compliance": {},
            }

            # HDR 형식 검출
            hdr_indicators = {
                "hdr10": ["pq", "smpte2084", "bt2084"],
                "hlg": ["hlg", "hybrid log-gamma", "arib"],
                "dolby_vision": ["dolby", "vision"],
            }

            detected_hdr = None
            for hdr_type, indicators in hdr_indicators.items():
                if any(
                    indicator in transfer_characteristics for indicator in indicators
                ):
                    detected_hdr = hdr_type
                    break

            if detected_hdr:
                result["compliance"]["hdr"] = {
                    "status": "pass",
                    "format": detected_hdr.upper().replace("_", " "),
                    "note": "HDR 메타데이터 확인 필요",
                }
            else:
                # Rec. 2020 색상 공간이지만 HDR 전송 특성이 없는 경우
                if "bt2020" in color_primaries or "rec2020" in color_primaries:
                    result["compliance"]["hdr"] = {
                        "status": "warning",
                        "reason": "Rec. 2020 색상 공간이지만 HDR 전송 특성 없음",
                    }
                else:
                    result["compliance"]["hdr"] = {
                        "status": "info",
                        "reason": "SDR 콘텐츠 (HDR 아님)",
                    }

            return result

        except Exception as e:
            return {"error": f"HDR 검사 중 오류: {str(e)}"}

    def calculate_compliance_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """종합 준수 점수 계산"""
        try:
            total_checks = 0
            passed_checks = 0
            warnings = 0
            failures = 0

            score_weights = {
                "container": 1,
                "codec": 3,
                "bitrate": 2,
                "color_system": 3,
                "bit_depth": 2,
                "resolution": 3,
                "frame_rate": 1,
                "hdr": 2,
            }

            weighted_score = 0
            max_weighted_score = sum(score_weights.values())

            detailed_results = {}

            for check_name, check_result in results.items():
                if "compliance" in check_result:
                    weight = score_weights.get(check_name, 1)
                    total_checks += 1

                    compliance = check_result["compliance"]
                    status_found = False

                    for key, value in compliance.items():
                        if isinstance(value, dict) and "status" in value:
                            status = value["status"]
                            status_found = True

                            if status == "pass":
                                passed_checks += 1
                                weighted_score += weight
                                detailed_results[check_name] = {
                                    "status": "pass",
                                    "weight": weight,
                                    "details": value,
                                }
                            elif status == "warning":
                                warnings += 1
                                weighted_score += weight * 0.5
                                detailed_results[check_name] = {
                                    "status": "warning",
                                    "weight": weight,
                                    "details": value,
                                }
                            elif status == "fail":
                                failures += 1
                                detailed_results[check_name] = {
                                    "status": "fail",
                                    "weight": weight,
                                    "details": value,
                                }
                            break

                    if not status_found:
                        detailed_results[check_name] = {
                            "status": "unknown",
                            "weight": weight,
                            "details": check_result,
                        }

            # 점수 계산
            percentage_score = (
                (weighted_score / max_weighted_score) * 100
                if max_weighted_score > 0
                else 0
            )

            # 등급 결정
            if percentage_score >= 90:
                grade = "A+ (우수)"
            elif percentage_score >= 80:
                grade = "A (양호)"
            elif percentage_score >= 70:
                grade = "B (보통)"
            elif percentage_score >= 60:
                grade = "C (미흡)"
            else:
                grade = "D (부족)"

            return {
                "overall_score": {
                    "percentage": round(percentage_score, 1),
                    "grade": grade,
                    "weighted_score": round(weighted_score, 1),
                    "max_score": max_weighted_score,
                },
                "summary": {
                    "total_checks": total_checks,
                    "passed": passed_checks,
                    "warnings": warnings,
                    "failures": failures,
                },
                "detailed_results": detailed_results,
                "recommendations": self._generate_recommendations(detailed_results),
            }

        except Exception as e:
            return {"error": f"점수 계산 중 오류: {str(e)}"}

    def _generate_recommendations(self, detailed_results: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        for check_name, result in detailed_results.items():
            if result["status"] == "fail":
                if check_name == "codec":
                    recommendations.append("HEVC (H.265) 또는 ProRes 코덱 사용 권장")
                elif check_name == "bit_depth":
                    recommendations.append("10-bit 이상의 비트 깊이 사용 필요")
                elif check_name == "resolution":
                    recommendations.append(
                        "UHD 4K (3840x2160) 또는 DCI 4K (4096x2160) 해상도 권장"
                    )
                elif check_name == "color_system":
                    recommendations.append("Rec. 2020 색상 공간 사용 권장")

            elif result["status"] == "warning":
                if check_name == "bitrate":
                    recommendations.append(
                        "스트리밍용: 10-20 Mbps, 프로덕션용: 320-480 Mbps 권장"
                    )
                elif check_name == "color_system":
                    recommendations.append("Rec. 2020 (Wide Color Gamut) 사용 권장")
                elif check_name == "hdr":
                    recommendations.append("HDR10, Dolby Vision, 또는 HLG 지원 권장")

        if not recommendations:
            recommendations.append("모든 기준을 충족합니다!")

        return recommendations

    def run_comprehensive_check(self, file_path: str, metadata: Dict) -> Dict[str, Any]:
        """종합 표준 준수 검사 실행"""
        try:
            print("DCI/OTT 표준 준수 검사 시작...")

            results = {}

            # 각 검사 항목 실행
            print("1. 컨테이너 형식 검사...")
            results["container"] = self.check_container_format(file_path, metadata)

            print("2. 비디오 코덱 검사...")
            results["codec"] = self.check_video_codec(metadata)

            print("3. 비트레이트 검사...")
            results["bitrate"] = self.check_bitrate(metadata)

            print("4. 색상 시스템 검사...")
            results["color_system"] = self.check_color_system(metadata)

            print("5. 비트 깊이 검사...")
            results["bit_depth"] = self.check_bit_depth(metadata)

            print("6. 해상도 및 화면비 검사...")
            results["resolution"] = self.check_resolution_and_aspect_ratio(metadata)

            print("7. 프레임레이트 검사...")
            results["frame_rate"] = self.check_frame_rate(metadata)

            print("8. HDR 지원 검사...")
            results["hdr"] = self.check_hdr_support(metadata)

            print("9. 종합 점수 계산...")
            compliance_score = self.calculate_compliance_score(results)

            return {
                "file_path": file_path,
                "standards_check": results,
                "compliance_score": compliance_score,
                "timestamp": metadata.get("analysis_timestamp", "unknown"),
            }

        except Exception as e:
            return {"error": f"종합 검사 중 오류: {str(e)}"}

    def print_results(self, results: Dict[str, Any]):
        """검사 결과 출력"""
        print("\n" + "=" * 80)
        print("DCI/OTT 표준 준수 검사 결과")
        print("=" * 80)

        if "error" in results:
            print(f"오류: {results['error']}")
            return

        file_name = Path(results["file_path"]).name
        print(f"파일: {file_name}")

        # 종합 점수
        if "compliance_score" in results:
            score = results["compliance_score"]["overall_score"]
            print(f"\n종합 점수: {score['percentage']}% ({score['grade']})")
            print(f"가중 점수: {score['weighted_score']}/{score['max_score']}")

            summary = results["compliance_score"]["summary"]
            print(
                f"검사 결과: 통과 {summary['passed']}, 경고 {summary['warnings']}, 실패 {summary['failures']}"
            )

        # 상세 결과
        print("\n" + "-" * 50)
        print("상세 검사 결과")
        print("-" * 50)

        check_names = {
            "container": "컨테이너 형식",
            "codec": "비디오 코덱",
            "bitrate": "비트레이트",
            "color_system": "색상 시스템",
            "bit_depth": "비트 깊이",
            "resolution": "해상도",
            "frame_rate": "프레임레이트",
            "hdr": "HDR 지원",
        }

        for check_key, check_name in check_names.items():
            if check_key in results.get("standards_check", {}):
                check_result = results["standards_check"][check_key]
                print(f"\n{check_name}:")

                if "error" in check_result:
                    print(f"  오류: {check_result['error']}")
                    continue

                # 컴플라이언스 상태 출력
                if "compliance" in check_result:
                    for sub_check, compliance in check_result["compliance"].items():
                        if isinstance(compliance, dict):
                            status = compliance.get("status", "unknown")

                            status_symbols = {
                                "pass": "✓",
                                "warning": "⚠",
                                "fail": "✗",
                                "info": "ℹ",
                            }

                            symbol = status_symbols.get(status, "?")
                            print(f"  {symbol} {sub_check}: ", end="")

                            if status == "pass":
                                level = compliance.get("level", "")
                                value = compliance.get(
                                    "value", compliance.get("standard", "")
                                )
                                if level:
                                    print(f"{value} ({level})")
                                else:
                                    print(f"{value}")

                                if "note" in compliance:
                                    print(f"    참고: {compliance['note']}")

                            elif status in ["warning", "fail"]:
                                reason = compliance.get("reason", "알 수 없는 이유")
                                print(f"{reason}")

                                if "supported" in compliance:
                                    print(
                                        f"    지원 형식: {', '.join(map(str, compliance['supported']))}"
                                    )

                            elif status == "info":
                                reason = compliance.get("reason", "")
                                print(f"{reason}")

        # 권장사항
        if (
            "compliance_score" in results
            and "recommendations" in results["compliance_score"]
        ):
            recommendations = results["compliance_score"]["recommendations"]
            print(f"\n" + "-" * 50)
            print("개선 권장사항")
            print("-" * 50)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

        print("\n" + "=" * 80)

    def save_results(self, results: Dict[str, Any], output_path: str):
        """검사 결과를 JSON 파일로 저장"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"DCI/OTT 표준 검사 결과가 저장되었습니다: {output_path}")
        except Exception as e:
            print(f"결과 저장 중 오류: {str(e)}")


def main():
    """테스트용 메인 함수"""
    print("DCI/OTT 표준 준수 검사기")
    print("META(PQL) / NETFLIX(VMAF) 품질평가 기준")

    # 비디오 메타데이터 분석기 import 필요
    try:
        import sys

        sys.path.append(str(Path(__file__).parent.parent / "analyzers"))
        from video_metadata_analyzer import VideoMetadataAnalyzer
    except ImportError:
        print("비디오 메타데이터 분석기를 찾을 수 없습니다.")
        print("video_metadata_analyzer.py가 src/analyzers/ 폴더에 있는지 확인하세요.")
        return

    video_file = input("검사할 비디오 파일 경로를 입력하세요: ").strip()

    if not video_file:
        print("비디오 파일 경로를 입력해주세요.")
        return

    try:
        # 1단계: 메타데이터 분석
        print("1단계: 비디오 메타데이터 분석 중...")
        metadata_analyzer = VideoMetadataAnalyzer()
        metadata = metadata_analyzer.analyze_complete(video_file)

        # 2단계: DCI/OTT 표준 검사
        print("2단계: DCI/OTT 표준 준수 검사 중...")
        standards_checker = DCIOTTStandardsChecker()

        # 타임스탬프 추가
        import datetime

        metadata["analysis_timestamp"] = datetime.datetime.now().isoformat()

        # 표준 검사 실행
        standards_results = standards_checker.run_comprehensive_check(
            video_file, metadata
        )

        # 결과 출력
        standards_checker.print_results(standards_results)

        # 결과 저장
        output_file = f"dci_ott_check_{Path(video_file).stem}.json"
        standards_checker.save_results(standards_results, output_file)

        # 메타데이터도 함께 저장
        combined_results = {
            "metadata_analysis": metadata,
            "standards_compliance": standards_results,
        }

        combined_output = f"complete_analysis_{Path(video_file).stem}.json"
        with open(combined_output, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)

        print(f"종합 분석 결과가 저장되었습니다: {combined_output}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
