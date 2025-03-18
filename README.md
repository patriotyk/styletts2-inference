# How to use with pytorch

```
from nltk.tokenize import word_tokenize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')


from styletts2_inference.models import StyleTTS2

model = StyleTTS2(hf_path='patriotyk/styletts2_ukrainian_multispeaker', device='cpu')
voice = model.compute_style('voices/Анастасія Павленко.wav')

text = 'Hello, how are you?'
ps = global_phonemizer.phonemize([text])
ps = word_tokenize(ps[0])

wav, _ = model(model.tokenizer.encode(' '.join(ps)), voice=voice)
soundfile.write('gennnerated.wav', wav.cpu().numpy(), 24000)

```