import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import ffmpeg
from pymediainfo import MediaInfo


class VideoMetadataAnalyzer:
    """
    비디오 파일의 메타데이터를 분석하는 클래스
    해상도, 코덱, 비트레이트 등의 정보를 추출합니다.
    """

    def __init__(self):
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

    def is_supported_format(self, file_path: str) -> bool:
        """지원되는 비디오 파일 형식인지 확인"""
        return Path(file_path).suffix.lower() in self.supported_formats

    def analyze_with_ffmpeg(self, file_path: str) -> Dict[str, Any]:
        """FFmpeg을 사용한 메타데이터 분석"""
        try:
            probe = ffmpeg.probe(file_path)

            # 비디오 스트림 찾기
            video_stream = None
            audio_streams = []

            for stream in probe["streams"]:
                if stream["codec_type"] == "video":
                    video_stream = stream
                elif stream["codec_type"] == "audio":
                    audio_streams.append(stream)

            if not video_stream:
                raise ValueError("비디오 스트림을 찾을 수 없습니다.")

            # 메타데이터 추출
            metadata = {
                "file_info": {
                    "filename": Path(file_path).name,
                    "file_size_mb": round(
                        os.path.getsize(file_path) / (1024 * 1024), 2
                    ),
                    "format": probe["format"]["format_name"],
                    "duration_seconds": float(probe["format"].get("duration", 0)),
                },
                "video": {
                    "codec_name": video_stream.get("codec_name"),
                    "codec_long_name": video_stream.get("codec_long_name"),
                    "width": int(video_stream.get("width", 0)),
                    "height": int(video_stream.get("height", 0)),
                    "aspect_ratio": video_stream.get("display_aspect_ratio"),
                    "pixel_format": video_stream.get("pix_fmt"),
                    "frame_rate": self._parse_frame_rate(
                        video_stream.get("r_frame_rate")
                    ),
                    "bit_rate": (
                        int(video_stream.get("bit_rate", 0))
                        if video_stream.get("bit_rate")
                        else None
                    ),
                    "profile": video_stream.get("profile"),
                    "level": video_stream.get("level"),
                },
                "audio": [],
            }

            # 오디오 스트림 정보
            for audio_stream in audio_streams:
                audio_info = {
                    "codec_name": audio_stream.get("codec_name"),
                    "codec_long_name": audio_stream.get("codec_long_name"),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                    "bit_rate": (
                        int(audio_stream.get("bit_rate", 0))
                        if audio_stream.get("bit_rate")
                        else None
                    ),
                    "language": audio_stream.get("tags", {}).get("language", "unknown"),
                }
                metadata["audio"].append(audio_info)

            return metadata

        except Exception as e:
            raise Exception(f"FFmpeg 분석 중 오류 발생: {str(e)}")

    def analyze_with_mediainfo(self, file_path: str) -> Dict[str, Any]:
        """MediaInfo를 사용한 상세 메타데이터 분석"""
        try:
            media_info = MediaInfo.parse(file_path)

            # 일반 정보
            general_track = None
            video_track = None
            audio_tracks = []

            for track in media_info.tracks:
                if track.track_type == "General":
                    general_track = track
                elif track.track_type == "Video":
                    video_track = track
                elif track.track_type == "Audio":
                    audio_tracks.append(track)

            if not video_track:
                raise ValueError("비디오 트랙을 찾을 수 없습니다.")

            metadata = {
                "file_info": {
                    "filename": Path(file_path).name,
                    "file_size_mb": round(
                        int(general_track.file_size or 0) / (1024 * 1024), 2
                    ),
                    "format": general_track.format,
                    "duration_ms": int(general_track.duration or 0),
                    "overall_bit_rate": int(general_track.overall_bit_rate or 0),
                },
                "video": {
                    "codec_name": video_track.format,
                    "codec_profile": video_track.format_profile,
                    "width": int(video_track.width or 0),
                    "height": int(video_track.height or 0),
                    "aspect_ratio": video_track.display_aspect_ratio,
                    "pixel_format": video_track.chroma_subsampling,
                    "frame_rate": float(video_track.frame_rate or 0),
                    "frame_count": int(video_track.frame_count or 0),
                    "bit_rate": int(video_track.bit_rate or 0),
                    "bit_depth": int(video_track.bit_depth or 0),
                    "color_space": video_track.color_space,
                    "color_primaries": video_track.color_primaries,
                    "transfer_characteristics": video_track.transfer_characteristics,
                    "matrix_coefficients": video_track.matrix_coefficients,
                },
                "audio": [],
            }

            # 오디오 트랙 정보
            for audio_track in audio_tracks:
                audio_info = {
                    "codec_name": audio_track.format,
                    "codec_profile": audio_track.format_profile,
                    "sample_rate": int(audio_track.sampling_rate or 0),
                    "channels": int(audio_track.channel_s or 0),
                    "channel_layout": audio_track.channel_layout,
                    "bit_rate": int(audio_track.bit_rate or 0),
                    "bit_depth": int(audio_track.bit_depth or 0),
                    "language": audio_track.language or "unknown",
                }
                metadata["audio"].append(audio_info)

            return metadata

        except Exception as e:
            raise Exception(f"MediaInfo 분석 중 오류 발생: {str(e)}")

    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """프레임 레이트 문자열을 숫자로 변환"""
        if not frame_rate_str:
            return 0.0

        try:
            if "/" in frame_rate_str:
                num, den = frame_rate_str.split("/")
                return round(float(num) / float(den), 3)
            else:
                return float(frame_rate_str)
        except:
            return 0.0

    def analyze_complete(self, file_path: str) -> Dict[str, Any]:
        """FFmpeg과 MediaInfo를 모두 사용한 종합 분석"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if not self.is_supported_format(file_path):
            raise ValueError(f"지원되지 않는 파일 형식입니다: {Path(file_path).suffix}")

        try:
            # FFmpeg과 MediaInfo 분석 결과 통합
            ffmpeg_result = self.analyze_with_ffmpeg(file_path)
            mediainfo_result = self.analyze_with_mediainfo(file_path)

            # 결과 통합 및 보완
            combined_result = {
                "analysis_tool": "Combined (FFmpeg + MediaInfo)",
                "file_path": str(Path(file_path).absolute()),
                "ffmpeg_data": ffmpeg_result,
                "mediainfo_data": mediainfo_result,
                "summary": self._create_summary(ffmpeg_result, mediainfo_result),
            }

            return combined_result

        except Exception as e:
            # 하나의 도구가 실패하면 다른 도구로만 분석
            print(f"종합 분석 실패, 개별 분석 시도: {str(e)}")

            try:
                result = self.analyze_with_ffmpeg(file_path)
                result["analysis_tool"] = "FFmpeg only"
                return result
            except:
                try:
                    result = self.analyze_with_mediainfo(file_path)
                    result["analysis_tool"] = "MediaInfo only"
                    return result
                except Exception as final_error:
                    raise Exception(f"모든 분석 도구 실패: {str(final_error)}")

    def _create_summary(
        self, ffmpeg_data: Dict, mediainfo_data: Dict
    ) -> Dict[str, Any]:
        """분석 결과 요약 생성"""
        video_ffmpeg = ffmpeg_data.get("video", {})
        video_mediainfo = mediainfo_data.get("video", {})

        return {
            "resolution": f"{video_ffmpeg.get('width', 0)}x{video_ffmpeg.get('height', 0)}",
            "codec": video_ffmpeg.get("codec_name", "Unknown"),
            "duration_seconds": ffmpeg_data.get("file_info", {}).get(
                "duration_seconds", 0
            ),
            "file_size_mb": ffmpeg_data.get("file_info", {}).get("file_size_mb", 0),
            "frame_rate": video_ffmpeg.get("frame_rate", 0),
            "bit_rate_kbps": round((video_ffmpeg.get("bit_rate", 0) or 0) / 1000, 2),
            "pixel_format": video_ffmpeg.get("pixel_format", "Unknown"),
            "color_space": video_mediainfo.get("color_space", "Unknown"),
            "audio_tracks_count": len(ffmpeg_data.get("audio", [])),
        }

    def save_analysis_result(self, result: Dict[str, Any], output_path: str):
        """분석 결과를 JSON 파일로 저장"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"분석 결과가 저장되었습니다: {output_path}")
        except Exception as e:
            raise Exception(f"결과 저장 중 오류 발생: {str(e)}")

    def print_summary(self, result: Dict[str, Any]):
        """분석 결과 요약 출력"""
        if "summary" in result:
            summary = result["summary"]
            print("\n=== 비디오 메타데이터 분석 결과 ===")
            print(f"파일명: {Path(result.get('file_path', '')).name}")
            print(f"해상도: {summary['resolution']}")
            print(f"코덱: {summary['codec']}")
            print(f"재생시간: {summary['duration_seconds']:.2f}초")
            print(f"파일크기: {summary['file_size_mb']}MB")
            print(f"프레임레이트: {summary['frame_rate']}fps")
            print(f"비트레이트: {summary['bit_rate_kbps']}kbps")
            print(f"픽셀포맷: {summary['pixel_format']}")
            print(f"색상공간: {summary['color_space']}")
            print(f"오디오트랙: {summary['audio_tracks_count']}개")
        else:
            print("요약 정보를 찾을 수 없습니다.")


def main():
    """테스트용 메인 함수"""
    analyzer = VideoMetadataAnalyzer()

    # 사용 예시
    video_file = input("분석할 비디오 파일 경로를 입력하세요: ").strip()

    if not video_file:
        print("샘플 비디오 파일이 필요합니다.")
        return

    try:
        print("비디오 분석 중...")
        result = analyzer.analyze_complete(video_file)

        # 결과 출력
        analyzer.print_summary(result)

        # JSON 파일로 저장
        output_file = f"{Path(video_file).stem}_metadata.json"
        analyzer.save_analysis_result(result, output_file)

    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
