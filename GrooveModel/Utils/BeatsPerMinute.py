from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

# Define BPM range explicitly
MIN_BPM = 20
MAX_BPM = 300
BPM_RESOLUTION = MAX_BPM - MIN_BPM

BPM_TOKEN_SIZE = BPM_RESOLUTION + SPECIAL_TOKEN_SIZE

def encode_bpm(bpm):
    if not (MIN_BPM <= bpm < MAX_BPM):
        raise ValueError(f"BPM {bpm} out of range ({MIN_BPM}-{MAX_BPM})")
    return (bpm - MIN_BPM) + SPECIAL_TOKEN_SIZE

def decode_bpm(token):
    bpm = (token - SPECIAL_TOKEN_SIZE) + MIN_BPM
    if not (MIN_BPM <= bpm < MAX_BPM):
        raise ValueError(f"Decoded BPM {bpm} out of range ({MIN_BPM}-{MAX_BPM})")
    return bpm
