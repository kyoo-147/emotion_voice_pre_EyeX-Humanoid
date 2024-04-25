# from speechbrain.pretrained.interfaces import foreign_class
# Nhập phiên bản mới đã được sửa đổi
from speechbrain.inference.interfaces import foreign_class
import gradio as gr
import os
import warnings

# Tắt các cảnh báo
warnings.filterwarnings("ignore")

# Chức năng lấy danh sách các tệp âm thanh trong thư mục 'rec/'
def get_audio_files_list(directory="rec"):
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        print("Thư mục 'rec' không tồn tại. Làm ơn chắc chắn là bạn đã nhập đúng địa chỉ.")
        return []

# Tải các tệp model đã được đào tạo
learner = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py", 
    classname="CustomEncoderWav2vec2Classifier"
)

# Xây dựng mảng chứa các trạng thái cảm xúc
emotion_dict = {
    'sad': 'Sad', 
    'hap': 'Happy',
    'ang': 'Anger',
    'fea': 'Fear',
    'sur': 'Surprised',
    'neu': 'Neutral'
}

def predict_emotion(selected_audio):
    file_path = os.path.join("rec", selected_audio)
    out_prob, score, index, text_lab = learner.classify_file(file_path)
    return emotion_dict[text_lab[0]]

# Nhận danh sách teẹp âm thanh 
audio_files_list = get_audio_files_list()
inputs = gr.Dropdown(label="Select Audio", choices=audio_files_list)

outputs = "text"
title = "Phát hiện cảm xúc lời nói ML"
description = "Mô hình tiền huấn luyện wav2vec 2.0 được hỗ trợ bởi Speechbrain trên tập dữ liệu IEMOCAP bằng Gradio."

interface = gr.Interface(fn=predict_emotion, inputs=inputs, outputs=outputs, title=title, description=description)
interface.launch()