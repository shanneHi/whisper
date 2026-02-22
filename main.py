import datetime
import os
import subprocess
import whisper
import opencc
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_whisper_turbo_supported():
    # openai-whisper 20240930+ adds large-v3-turbo / turbo model names
    try:
        from importlib import metadata
    except Exception:
        return
    try:
        version_str = metadata.version("openai-whisper")
    except Exception:
        return
    m = re.search(r"(\d{8})", version_str)
    if not m:
        return
    try:
        version_num = int(m.group(1))
    except ValueError:
        return
    if version_num < 20240930:
        print(
            f"openai-whisper 版本过旧({version_str})，不支持 large-v3-turbo。将回退到 large-v3。"
        )
        return False
    return True


def _ln_fp16_fix(self, x):
    """修复 Whisper 在 FP16 下 LayerNorm 权重 / 偏置 dtype 不匹配的问题"""
    return F.layer_norm(
        x.float(),
        self.normalized_shape,
        self.weight.float() if self.weight is not None else None,
        self.bias.float() if self.bias is not None else None,
        self.eps,
    ).type_as(x)

# Monkey-patch 所有 LayerNorm.forward
nn.LayerNorm.forward = _ln_fp16_fix


def extract_audio_from_flv(input_folder, output_folder):
    # ① 遍历 input_folder 下所有 .flv
    # ② 调用 ffmpeg 提取音轨为 .mp3
    # ③ 输出到 output_folder
    # ④ 提取完毕后删除原 .flv
    print(f"提取音频文件到 {output_folder}...")

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".flv"):
            # 构建输入文件的完整路径
            input_file_path = os.path.join(input_folder, file_name)

            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.mp3")

            # 使用ffmpeg提取音频
            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_file_path,
                "-vn",
                "-acodec",
                "libmp3lame",
                output_file_path,
            ]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError:
                print(f"音频提取失败: {input_file_path}")
                continue
            print(f"音频文件已提取到 {output_file_path}")
            try:
                os.remove(input_file_path)
                print(f"删除文件 {file_name}")
            except OSError:
                print(f"删除失败: {file_name}")
    print("音频文件提取完成。")
    print("删除完成。")

# 设置输入文件夹和输出文件夹的路径
input_video_folder_path =  r"D:\douyin"
MP3_temp_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Input")
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "Output")
save_MP3_folder = os.path.join(os.path.expanduser("~"), "Desktop", "MP3")

# 确保输出文件夹存在，如果不存在则创建它
os.makedirs(MP3_temp_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(save_MP3_folder, exist_ok=True)


# 调用函数提取音频
extract_audio_from_flv(input_video_folder_path, MP3_temp_folder)





def _unique_target_path(folder, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    if not os.path.exists(candidate):
        return candidate
    index = 1
    while True:
        candidate = os.path.join(folder, f"{base}_{index}{ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


# 获取输入文件夹下所有音频文件
audio_files = sorted(
    f for f in os.listdir(MP3_temp_folder) if f.endswith(".mp3") or f.endswith(".wav")
)
if not audio_files:
    print("没有找到可转录的音频文件。")
else:
    # 加载 Whisper 模型到可用设备，并在 CUDA 时使用 FP16 以提升推理速度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _ensure_whisper_turbo_supported():
        model = whisper.load_model("large-v3-turbo", device=device)
    else:
        model = whisper.load_model("large-v3", device=device)
    if device == "cuda":
        model = model.half()
    converter = opencc.OpenCC("t2s")

    # 打印开始时间
    start_time = datetime.datetime.now()
    print(f"开始时间: {start_time}")

    for audio_file in audio_files:
        input_path = os.path.join(MP3_temp_folder, audio_file)

        # 转录音频文件
        result = whisper.transcribe(
            model,
            input_path,
            language="zh",
            temperature=[0.0, 0.2],
            beam_size=5,
            condition_on_previous_text=False,
        )

        # 定义保存文本文件的路径和文件名
        output_file = os.path.splitext(audio_file)[0] + ".txt"
        output_path = os.path.join(output_folder, output_file)

        # 将简体中文文本写入到文件
        with open(output_path, "w", encoding="utf-8") as file:
            for seg in result["segments"]:
                line = converter.convert(seg["text"].strip())
                if not line:
                    continue
                if not re.search(r"[。！？…]$", line):
                    line += "。"
                file.write(line + "\n")
        # 时间
        now = datetime.datetime.now()
        print("当前时间：")
        print(now)
        # 时间差
        time = now - start_time
        print("时间差：")
        print(time)
        start_time = now

        print(f"音频文件 {audio_file} 已转录。")
        print(f"文本已保存到 {output_path}")

    # 把音频文件转移到save_MP3_folder文件夹
    for audio_file in audio_files:
        input_path = os.path.join(MP3_temp_folder, audio_file)
        output_path = _unique_target_path(save_MP3_folder, audio_file)
        os.rename(input_path, output_path)
        print(f"音频文件已保存到 {output_path}")




# 打印结束时间
end_time = datetime.datetime.now()
print(f"结束时间: {end_time}")
if audio_files:
    print("所有音频文件均已转录并保存。")
else:
    print("没有音频文件需要处理。")
