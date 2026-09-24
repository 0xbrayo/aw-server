"""Bounded query results with dependencies on the actual datastore reads."""

import copy
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from time import monotonic
from typing import Any, List, Optional, Tuple

Read = Tuple[str, Optional[datetime], Optional[datetime]]


def _size(value, seen=None):
    """Account for retained Python objects, including keys and nested payloads."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set)):
        size += sum(_size(v, seen) for v in value)
    return size


@dataclass
class Change:
    span: Optional[Tuple[datetime, datetime]] = None


@dataclass
class Entry:
    result: Any
    reads: List[Read]
    metadata: bool
    expires: float
    size: int


class QueryCache:
    def __init__(self, max_entries=128, max_bytes=8 * 1024 * 1024, ttl=300):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl = ttl
        self.entries: OrderedDict = OrderedDict()
        self.bytes = 0
        self.revision = 0
        self.writers = 0
        self.lock = RLock()

    def _remove(self, key):
        self.bytes -= self.entries.pop(key).size

    def lookup(self, key):
        with self.lock:
            for expired in [
                k for k, e in self.entries.items() if e.expires <= monotonic()
            ]:
                self._remove(expired)
            if self.writers:
                return False, None, None
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                return True, copy.deepcopy(entry.result), self.revision
            return False, None, self.revision

    def store(self, key, result, tracker, revision):
        if revision is None or not tracker.cacheable or self.max_entries <= 0:
            return
        # Do the potentially expensive copy outside the cache lock.
        size = _size((key, result, tracker.reads))
        if size > self.max_bytes:
            return
        snapshot = copy.deepcopy(result)
        with self.lock:
            if self.writers or self.revision != revision:
                return
            if key in self.entries:
                self._remove(key)
            while self.entries and (
                len(self.entries) >= self.max_entries
                or self.bytes + size > self.max_bytes
            ):
                self._remove(next(iter(self.entries)))
            self.entries[key] = Entry(
                snapshot,
                list(tracker.reads),
                tracker.metadata,
                monotonic() + self.ttl,
                size,
            )
            self.bytes += size

    @contextmanager
    def mutation(self, bucket_id, metadata=False):
        change = Change()
        with self.lock:
            self.writers += 1
            self.revision += 1
        try:
            yield change
        except BaseException:
            # A failed bulk operation can still have made partial changes.
            change.span = None
            raise
        finally:
            with self.lock:
                for key, entry in list(self.entries.items()):
                    affected = metadata and entry.metadata
                    for bid, start, end in entry.reads:
                        if bid != bucket_id:
                            continue
                        if metadata or change.span is None:
                            affected = True
                        else:
                            lo, hi = sorted(change.span)
                            affected |= (end is None or lo <= end) and (
                                start is None or start <= hi
                            )
                    if affected:
                        self._remove(key)
                self.writers -= 1
                self.revision += 1


@dataclass
class ReadTracker:
    datastore: Any
    reads: List[Read] = field(default_factory=list)
    metadata: bool = False
    cacheable: bool = True

    def buckets(self):
        self.metadata = True
        return self.datastore.buckets()

    def __getitem__(self, bucket_id):
        return ReadBucket(self.datastore[bucket_id], self, bucket_id)

    def __getattr__(self, name):
        # Custom query functions using other datastore operations must not cache
        # a result whose dependencies we cannot describe.
        self.cacheable = False
        return getattr(self.datastore, name)


class ReadBucket:
    def __init__(self, bucket, tracker, bucket_id):
        self.bucket = bucket
        self.tracker = tracker
        self.bucket_id = bucket_id

    def _record(self, starttime, endtime):
        # Bucket.get rounds start down and end up to millisecond boundaries.
        padding = timedelta(milliseconds=1)
        try:
            start = starttime - padding if starttime else None
        except OverflowError:
            start = None
        try:
            end = endtime + padding if endtime else None
        except OverflowError:
            end = None
        self.tracker.reads.append(
            (
                self.bucket_id,
                start,
                end,
            )
        )

    def get(self, limit=-1, starttime=None, endtime=None):
        self._record(starttime, endtime)
        return self.bucket.get(limit, starttime, endtime)

    def get_eventcount(self, starttime=None, endtime=None):
        self._record(starttime, endtime)
        return self.bucket.get_eventcount(starttime, endtime)

    def metadata(self):
        self.tracker.metadata = True
        return self.bucket.metadata()

    def __getattr__(self, name):
        self.tracker.cacheable = False
        return getattr(self.bucket, name)
