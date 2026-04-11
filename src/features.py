"""
features.py — Feature engineering pipeline (column names confirmed from EDA).
"""
LIVE_ELECTRODES = ['F3','F4','C3','C4']
EEG_BANDS       = ['Alpha','Gamma','HighBeta','LowBeta','Theta']
EEG_COLS        = [f'{e}{b}' for e in LIVE_ELECTRODES for b in EEG_BANDS]
SQ_COLS         = [f'{e}SignalQuality' for e in LIVE_ELECTRODES]
GAME_COLS       = ['PlayerPositionY','Morale','LevelProgress','OngoingAsteroid']
PROTO_COLS      = ['TangentCoefficient','TranslationCoefficient','Baseline',
                   'MinBaseline','MiddleBaseline','MaxBaseline',
                   'Percentile','MinPercentile','MiddlePercentile','MaxPercentile',
                   'ValueFiltered']
TARGET_COL      = 'ProtocolValue'
SUBSESSION_COL  = 'subsession'
DEAD_ELECTRODES = ['AF3','AF4','F7','F8','FC5','FC6','Fp1','Fp2',
                   'O1','O2','Oz','P7','P8','Pz','T7','T8']
