## How to use with pytorch

```
import soundfile
from nltk.tokenize import word_tokenize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')


from styletts2_inference.models import StyleTTS2

model = StyleTTS2(hf_path='patriotyk/StyleTTS2-LibriTTS', device='cpu')

text = 'Hello, how are you?'
ps = global_phonemizer.phonemize([text])
ps = ' '.join(word_tokenize(ps[0]))
tokens = model.tokenizer.encode(ps)

style = model.predict_style_multi('prompt.wav', tokens)

wav = model(tokens, s_prev=style)
soundfile.write('gennnerated.wav', wav.cpu().numpy(), 24000)

```

## How to use with onnx

First you need to export model to onnx format using included script `export_onnx.py`. This script will generate
`styletts2.onnx` file in the current directory.

Then you can infer it with the following code:
```
import onnxruntime
import soundfile
import numpy
from styletts2_inference.models import StyleTTS2Tokenizer

from nltk.tokenize import word_tokenize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

text = 'Hello, how are you?'
ps = global_phonemizer.phonemize([text])
ps = ' '.join(word_tokenize(ps[0]))

styletts2_session = onnxruntime.InferenceSession("styletts.onnx")
tokenizer = StyleTTS2Tokenizer(hf_path='patriotyk/StyleTTS2-LJSpeech')
tokens = tokenizer.encode(ps)

wav, _ = styletts2_session.run(None, {'tokens': tokens.numpy(),
                                'speed': [1.0],
                                's_prev': model.predict_style_single(tokens)
                                })
soundfile.write('gennnerated_onnx.wav', wav, 24000)


```

For multispeaker, you have to generate `style` vector from audio file and pass it to onnx session. You can do it using pytorch model using method `predict_style_multi` or `predict_style_single`.