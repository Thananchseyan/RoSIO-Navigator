from abc import ABC, abstractmethod

"""
We use two levels of hierarchy for flexible data loading pipeline:
  - Sequence: Read the sequence from file and compute per-frame feature and target.
  - Dataset: subclasses of PyTorch's Dataset class. It has three roles:
      1. Create a Sequence instance internally to load data and compute feature/target.
      2. Apply post processing, e.g. smoothing or truncating, to the loaded sequence.
      3. Define how to extract samples from the sequence.


To define a new dataset for training/testing:
  1. Subclass CompiledSequence class. Load data and compute feature/target in "load()" function.
  2. Subclass the PyTorch Dataset. In the constructor, use the custom CompiledSequence class to load data. You can also
     apply additional processing to the raw sequence, e.g. smoothing or truncating. Define how to extract samples from 
     the sequences by overriding "__getitem()__" function.
  3. If the feature/target computation are expensive, consider using "load_cached_sequence" function.

Please refer to GlobalSpeedSequence and DenseSequenceDataset in data_global_speed.py for reference. 
"""


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass

    def get_meta(self):
        return "No info available"


def load_cached_sequences(seq_type, imu_sequence, **kwargs):
    features_all, targets_all, aux_all = [], [], []
    seq = seq_type(imu_sequence, **kwargs)
    feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
    features_all.append(feat)
    targets_all.append(targ)
    aux_all.append(aux)
    return features_all, targets_all, aux_all