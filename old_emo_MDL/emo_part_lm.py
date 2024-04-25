from speechbrain.inference.diarization import Speech_Emotion_Diarization

classifier = Speech_Emotion_Diarization.from_hparams(
    source="speechbrain/emotion-diarization-wavlm-large"
)

# 
diary = classifier.diarize_file("example.wav")
print(diary)

# {
#    'speechbrain/emotion-diarization-wavlm-large/example.wav':
#       [
#          {'start': 0.0, 'end': 1.94, 'emotion': 'n'}, # n -> neutral
#          {'start': 1.94, 'end': 4.48, 'emotion': 'h'} # h -> happy
#       ]
# }

diary = classifier.diarize_file("example_sad.wav")
print(diary)

# {
#    'speechbrain/emotion-diarization-wavlm-large/example_sad.wav':
#        [
#          {'start': 0.0, 'end': 3.54, 'emotion': 's'}, # s -> sad
#          {'start': 3.54, 'end': 5.26, 'emotion': 'n'} # n -> neutral
#        ]
# }
