"""
datacore -- a standalone data subsystem: prepares a raw text/token corpus into a pretokenized,
packed, multipart on-disk dataset, and reads it back as flexible-batch-size, DDP-shardable,
exactly-resumable batches. Knows nothing about a host application's dataset identity (which URL,
which shard count, which task mixture) -- that's the host's job, sitting on top and producing a
source for datacore to prepare. See datacore/docs/architecture.md for the full contract.

DataManager is the one entrypoint; Tokenizer/CharTokenizer, Packer/BestFitCropPacker/
BestFitPadPacker/EncodedDoc/PackedRow, TextSource/TokenSource/ParquetDirectorySource,
DatasetStore/FileSystemDatasetStore, Dataset/DatasetInfo are the value types and protocols that
cross its boundary. Everything else (writer internals, reader internals) is internal.

Importing this package triggers no side effects (unlike modelcore's @register_component catalog)
-- there is no registry here to populate.
"""
from datacore.manager import DataManager
from datacore.packing import BestFitCropPacker, BestFitPadPacker, EncodedDoc, Packer, PackedRow
from datacore.reader import Dataset, DatasetInfo
from datacore.sources import ParquetDirectorySource, TextSource, TokenSource
from datacore.store import DatasetStore, FileSystemDatasetStore
from datacore.tokenizer import CharTokenizer, Tokenizer

__all__ = [
    "DataManager",
    "Tokenizer", "CharTokenizer",
    "Packer", "BestFitCropPacker", "BestFitPadPacker", "EncodedDoc", "PackedRow",
    "TextSource", "TokenSource", "ParquetDirectorySource",
    "DatasetStore", "FileSystemDatasetStore",
    "Dataset", "DatasetInfo",
]
