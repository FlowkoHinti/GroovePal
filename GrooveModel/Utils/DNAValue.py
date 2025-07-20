from enum import IntEnum, auto

from GrooveModel.Utils.SpecialTokens import SPECIAL_TOKEN_SIZE


class InstrumentValues(IntEnum):
    Rest = 0
    Kick = 1 << 0
    Snare = 1 << 1
    Toms = 1 << 2
    HiHat = 1 << 3
    Ride = 1 << 4
    Crash = 1 << 5


class RemappedInstrumentValues(IntEnum):
    Rest = SPECIAL_TOKEN_SIZE
    Kick = auto()
    Snare = auto()
    Toms = auto()
    HiHat = auto()
    Ride = auto()
    Crash = auto()


DNA_VALUE_TOKEN_SIZE = len(InstrumentValues) + SPECIAL_TOKEN_SIZE


def get_dna_instruments_list(value) -> list[InstrumentValues]:
    """
    Get the instruments values from the dna value.
    :param value: composite value
    :return: list of instrument values
    """
    instruments = []
    for instrument in InstrumentValues:
        if value & instrument.value:
            instruments.append(instrument.value)
    return instruments

def encode_instrument(value) -> RemappedInstrumentValues:
    """
    Get the remapped instruments values from the dna value.
    :param value: value of a single instrument
    :return: list of remapped instrument values
    """
    dna_instrument = InstrumentValues(value).name
    return RemappedInstrumentValues[dna_instrument]

def decode_instrument(remapped_value) -> InstrumentValues:
    """
    Get the original instrument value from a remapped instrument value.
    :param remapped_value: a value from RemappedInstrumentValues
    :return: original instrument value as int
    """
    # Reverse the RemappedInstrumentValues mapping
    instrument = RemappedInstrumentValues(remapped_value).name
    return InstrumentValues[instrument]

def dna_to_instruments_strings(value) -> list[str]:
    """
    Get the instruments from the dna value.
    :param value: The value to get the instruments from.
    :return: The instruments.
    """
    instruments = []
    for instrument in InstrumentValues:
        if value & instrument.value:
            instruments.append(instrument.name)
    return instruments


def instruments_strings_to_dna(instruments: list[str]) -> int:
    """
    Get the dna value from the instruments.
    :param instruments: The instruments to get the dna value from.
    :return: The dna value.
    """
    value = 0
    for instrument in instruments:
        value |= InstrumentValues[instrument].value
    return value
