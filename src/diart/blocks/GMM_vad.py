import torch
import struct
import webrtcvad
import soundfile as sf

VAD = webrtcvad.Vad()

def measure_vad(wav,sr=16000, vad_object=None):
    """uses webrtc vad to measure vad probability over a waveform, segment size is 30ms

    Args:
        wav (np.ndarray or torch.Tensor): waveform to measure vad on
        sr (int, optional): sampling rate. Defaults to 16000.

    Returns:
        float: probability of voice activity in the waveform
    """
    vad_object = vad_object or VAD
    wav = torch.tensor(wav).unsqueeze(0)
    frames = wav.unfold(dimension=1, size=int(sr*(30/1000)), step=int(sr*(1/1000)))
    total_vad = 0
    for i in range(frames.shape[1]):
        frame = frames[:,i,:].squeeze(0).tolist()
        frame_bytes = b''.join(struct.pack('h',samp) for samp in frame)
        vad = vad_object.is_speech(buf=frame_bytes,sample_rate=sr)
        total_vad += vad/frames.shape[1]
    return total_vad

if __name__ == '__main__':
    STEP = 0.5
    CHANNELS = 1 
    RATE = 16000
    SAMPLE_RATE = 16000
    FRAME_LEN = STEP
    THRESHOLD = 0.5
    # print(cfg)
    CHUNK_SIZE = int(STEP * RATE)
    
    
    wf, fs = sf.read('/home/storage01/diart/rttm_ariel_2/wav_data/3_99.wav',dtype='int16')
    data = wf[:CHUNK_SIZE]
    wf = wf[CHUNK_SIZE:]
    # data = wf.readframes(CHUNK_SIZE)
    vad= []
    while len(data):
        vad.append(measure_vad(data))
        data = wf[:CHUNK_SIZE]
        wf = wf[CHUNK_SIZE:]
        

    import matplotlib.pyplot as plt
    
    wf, fs = sf.read('/home/storage01/diart/rttm_ariel_2/wav_data/3_99.wav',dtype='int16')
    
    plt.plot([i*1/fs for i in range(len(wf))],wf)
    plt.plot([i*STEP for i in range(len(vad))],[v*max(wf) for v in vad])
    
    plt.savefig('vad.png')