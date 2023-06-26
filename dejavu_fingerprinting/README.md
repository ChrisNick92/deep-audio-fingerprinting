# dejavu

This folder contains some python wrappers around the open source library <a href="https://github.com/worldveil/dejavu/tree/master">dejavu</a>. Dejavu is an open source library that uses the audio fingerprinting technique for recognizing audio fragments captured from user's microphone. This approach is also used by well-established services such Shazam. More information, about this approach can be found in host url of the library as well as in a very nice <a href="https://willdrevo.com/fingerprinting-and-audio-recognition-with-python/">blog</a> made by the creator.

These wrappers are only used to compare an audio fingerprinting system, such as dejavu, with the modern deep-audio-fingerprinting techniques which where initially introduced in ascending chronological order in the papers:

[1] <a href="https://arxiv.org/pdf/1711.10958.pdf"> Now Playing: Continuous low-power music recognition</a> by Google's research team.

[2] <a href="https://arxiv.org/pdf/2010.11910.pdf"> Neural Audio Fingerprint for high-specific audio retrieval based on contrastive learning</a>.

[3] <a href="https://arxiv.org/abs/2210.08624"> Attention-Based Audio Embeddings for Query-by-Example</a>.