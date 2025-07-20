from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE

#TODO: BPM normalization (will do 0 - 300 for right now)
BPM_RESOLUTION = 300
BPM_TOKEN_SIZE = BPM_RESOLUTION + SPECIAL_TOKEN_SIZE

def encode_bpm(bpm):
    return bpm + SPECIAL_TOKEN_SIZE

def decode_bpm(bpm):
    return bpm - SPECIAL_TOKEN_SIZE
