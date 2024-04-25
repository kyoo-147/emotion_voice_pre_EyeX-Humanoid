# Phiên bản cũ sẽ có tên pretrained
# from speechbrain.pretrained.interfaces import foreign_class
# import gradio as gr
# Đối với new version, module pretrained sẽ được thay đổi tên thành inference
from speechbrain.inference.interfaces import foreign_class
import warnings
# Dis các warnings
warnings.filterwarnings("ignore")

# Tải mô hình được đào tạo truớc để nhận task reg emo
learner = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py", 
    classname="CustomEncoderWav2vec2Classifier"
)

# Mảng lưu trữ trạng tái cảm xúc
emotion_dict = {
    'sad': 'Sad', 
    'hap': 'Happy',
    'ang': 'Anger',
    'fea': 'Fear',
    'sur': 'Surprised',
    'neu': 'Neutral'
}

def predict_emotion(audio):
    out_prob, score, index, text_lab = learner.classify_file(audio)
    return emotion_dict[text_lab[0]]

# Put tệp âm thanh
audio = "rec/rec_sad.wav"
print("Predict: ", predict_emotion(audio))
