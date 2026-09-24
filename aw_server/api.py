import copy
import functools
import json
import logging
from datetime import datetime
from pathlib import Path
from socket import gethostname
from threading import RLock
from contextlib import closing
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from uuid import uuid4

import iso8601
from aw_core.dirs import get_data_dir
from aw_core.log import get_log_file_path
from aw_core.models import Event
from aw_query import query2
from aw_query.exceptions import QueryException
from aw_transform import heartbeat_merge

from .__about__ import __version__
from .exceptions import NotFound
from .profile import profile_from_env
from .query_cache import QueryCache, ReadTracker
from .settings import Settings

logger = logging.getLogger(__name__)


def get_device_id() -> str:
    path = Path(get_data_dir("aw-server")) / "device_id"
    if path.exists():
        with open(path) as f:
            return f.read()
    else:
        uuid = str(uuid4())
        with open(path, "w") as f:
            f.write(uuid)
        return uuid


def check_bucket_exists(f):
    @functools.wraps(f)
    def g(self, bucket_id, *args, **kwargs):
        # Datastore caches bucket handles and invalidates them on deletion.
        # Reuse that lookup instead of loading every bucket's metadata per call.
        try:
            self.db[bucket_id]
        except KeyError:
            raise NotFound(
                "NoSuchBucket", f"There's no bucket named {bucket_id}"
            ) from None
        return f(self, bucket_id, *args, **kwargs)

    return g


def bucket_mutation(metadata=False):
    """Serialize writes with heartbeat merging and invalidate dependent queries."""

    def decorate(f):
        @functools.wraps(f)
        def wrapped(self, bucket_id, *args, **kwargs):
            with self._write_lock, self.query_cache.mutation(bucket_id, metadata):
                self.last_event.pop(bucket_id, None)
                return f(self, bucket_id, *args, **kwargs)

        return wrapped

    return decorate


class ServerAPI:
    def __init__(self, db, testing) -> None:
        self.db = db
        self.settings = Settings(testing)
        self.testing = testing
        self.profile = profile_from_env(testing=testing)
        self.last_event = {}  # type: dict
        self._write_lock = RLock()
        self.query_cache = QueryCache()

    def get_info(self) -> Dict[str, Any]:
        """Get server info"""
        payload = {
            "hostname": gethostname(),
            "version": __version__,
            "testing": self.testing,
            "device_id": get_device_id(),
            "profile": self.profile,
        }
        return payload

    def get_buckets(self) -> Dict[str, Dict]:
        """Get dict {bucket_name: Bucket} of all buckets"""
        logger.debug("Received get request for buckets")
        return self.db.buckets(include_last_updated=True)

    @check_bucket_exists
    def get_bucket_metadata(self, bucket_id: str) -> Dict[str, Any]:
        """Get metadata about bucket."""
        bucket = self.db[bucket_id]
        return bucket.metadata()

    @check_bucket_exists
    def export_bucket(self, bucket_id: str) -> Dict[str, Any]:
        """Export a bucket to a dataformat consistent across versions, including all events in it."""
        bucket = self.get_bucket_metadata(bucket_id)
        bucket["events"] = self.get_events(bucket_id, limit=-1)
        # Scrub event IDs
        for event in bucket["events"]:
            del event["id"]
        return bucket

    def export_all(self) -> Dict[str, Any]:
        """Exports all buckets and their events to a format consistent across versions"""
        buckets = self.db.buckets()
        exported_buckets = {}
        for bid in buckets.keys():
            exported_buckets[bid] = self.export_bucket(bid)
        return exported_buckets

    def stream_export(self, bucket_id=None):
        # Validate before returning the generator so a missing bucket is a 404,
        # not an exception after the HTTP response has started.
        if bucket_id is not None:
            buckets = {bucket_id: self.get_bucket_metadata(bucket_id)}
        else:
            buckets = self.db.buckets()

        def generate():
            yield '{"buckets":{'
            for index, (bid, metadata) in enumerate(buckets.items()):
                if index:
                    yield ","
                yield json.dumps(bid) + ":"
                yield json.dumps(metadata)[:-1] + ',"events":['
                with closing(self.db[bid].iter_events()) as events:
                    for event_index, event in enumerate(events):
                        if event_index:
                            yield ","
                        payload = event.to_json_dict()
                        payload.pop("id", None)
                        yield json.dumps(payload)
                yield "]}"
            yield "}}"

        def buffered():
            # Avoid a socket write for every separator/small event while retaining
            # only a bounded batch (plus the largest individual event).
            with closing(generate()) as fragments:
                batch = []
                size = 0
                for fragment in fragments:
                    batch.append(fragment)
                    size += len(fragment)
                    if size >= 64 * 1024:
                        yield "".join(batch)
                        batch = []
                        size = 0
                if batch:
                    yield "".join(batch)

        return buffered()

    def import_bucket(self, bucket_data: Any):
        bucket_id = bucket_data["id"]
        with self._write_lock, self.query_cache.mutation(bucket_id, metadata=True):
            self.last_event.pop(bucket_id, None)
            logger.info(f"Importing bucket {bucket_id}")

            # TODO: Check that bucket doesn't already exist
            self.db.create_bucket(
                bucket_id,
                type=bucket_data["type"],
                client=bucket_data["client"],
                hostname=bucket_data["hostname"],
                created=(
                    bucket_data["created"]
                    if isinstance(bucket_data["created"], datetime)
                    else iso8601.parse_date(bucket_data["created"])
                ),
            )

            # scrub IDs from events
            # (otherwise causes weird bugs with no events seemingly imported when importing events exported from aw-server-rust, which contains IDs)
            for event in bucket_data["events"]:
                if "id" in event:
                    del event["id"]

            self.create_events(
                bucket_id,
                [
                    Event(**e) if isinstance(e, dict) else e
                    for e in bucket_data["events"]
                ],
            )

    def import_all(self, buckets: Dict[str, Any]):
        for bid, bucket in buckets.items():
            self.import_bucket(bucket)

    @bucket_mutation(metadata=True)
    def create_bucket(
        self,
        bucket_id: str,
        event_type: str,
        client: str,
        hostname: str,
        created: Optional[datetime] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a bucket.

        If hostname is "!local", the hostname and device_id will be set from the server info.
        This is useful for watchers which are known/assumed to run locally but might not know their hostname (like aw-watcher-web).

        Returns True if successful, otherwise false if a bucket with the given ID already existed.
        """
        if created is None:
            created = datetime.now()
        if self.db.has_bucket(bucket_id):
            return False
        if hostname == "!local":
            info = self.get_info()
            if data is None:
                data = {}
            hostname = info["hostname"]
            data["device_id"] = info["device_id"]
        self.db.create_bucket(
            bucket_id,
            type=event_type,
            client=client,
            hostname=hostname,
            created=created,
            data=data,
        )
        return True

    @bucket_mutation(metadata=True)
    @check_bucket_exists
    def update_bucket(
        self,
        bucket_id: str,
        event_type: Optional[str] = None,
        client: Optional[str] = None,
        hostname: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update bucket metadata"""
        self.db.update_bucket(
            bucket_id,
            type_id=event_type,
            client=client,
            hostname=hostname,
            data=data,
        )
        return None

    @bucket_mutation(metadata=True)
    @check_bucket_exists
    def delete_bucket(self, bucket_id: str) -> None:
        """Delete a bucket"""
        self.db.delete_bucket(bucket_id)
        logger.debug("Deleted bucket '%s'", bucket_id)
        return None

    @check_bucket_exists
    def get_event(
        self,
        bucket_id: str,
        event_id: int,
    ) -> Optional[Event]:
        """Get a single event from a bucket"""
        logger.debug(
            "Received get request for event %s in bucket '%s'", event_id, bucket_id
        )
        event = self.db[bucket_id].get_by_id(event_id)
        return event.to_json_dict() if event else None

    @check_bucket_exists
    def get_events(
        self,
        bucket_id: str,
        limit: int = -1,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Event]:
        """Get events from a bucket"""
        logger.debug("Received get request for events in bucket '%s'", bucket_id)
        if limit is None:  # Let limit = None also mean "no limit"
            limit = -1
        events = [
            event.to_json_dict() for event in self.db[bucket_id].get(limit, start, end)
        ]
        return events

    @bucket_mutation(metadata=False)
    @check_bucket_exists
    def create_events(self, bucket_id: str, events: List[Event]) -> List[Event]:
        """Create events for a bucket. Can handle both single events and multiple ones.

        Always returns a list of inserted events (matching aw-server-rust behavior).
        For single events, the returned event includes the server-assigned ID.
        For bulk inserts, returns empty list (events may not have IDs without a response-SQL roundtrip).
        """
        if len(events) == 1:
            # Pass as single Event so Bucket.insert uses insert_one (returns Event with ID)
            inserted = self.db[bucket_id].insert(events[0])
            return [inserted]
        else:
            self.db[bucket_id].insert(events)
            return []

    @check_bucket_exists
    def get_eventcount(
        self,
        bucket_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> int:
        """Get eventcount from a bucket"""
        logger.debug("Received get request for eventcount in bucket '%s'", bucket_id)
        return self.db[bucket_id].get_eventcount(start, end)

    @bucket_mutation(metadata=False)
    @check_bucket_exists
    def delete_event(self, bucket_id: str, event_id) -> bool:
        """Delete a single event from a bucket"""
        return self.db[bucket_id].delete(event_id)

    def heartbeat(self, bucket_id: str, heartbeat: Event, pulsetime: float) -> Event:
        """
        Heartbeats are useful when implementing watchers that simply keep
        track of a state, how long it's in that state and when it changes.
        A single heartbeat always has a duration of zero.

        If the heartbeat was identical to the last (apart from timestamp), then the last event has its duration updated.
        If the heartbeat differed, then a new event is created.

        Such as:
         - Active application and window title
           - Example: aw-watcher-window
         - Currently open document/browser tab/playing song
           - Example: wakatime
           - Example: aw-watcher-web
           - Example: aw-watcher-spotify
         - Is the user active/inactive?
           Send an event on some interval indicating if the user is active or not.
           - Example: aw-watcher-afk

        Inspired by: https://wakatime.com/developers#heartbeats
        """
        with self._write_lock, self.query_cache.mutation(bucket_id) as change:
            event = self._heartbeat(bucket_id, heartbeat, pulsetime)
            # Invalidate every interval touched by the event, including the old
            # duration when a heartbeat is merged. Other days stay cached.
            change.span = (event.timestamp, event.timestamp + event.duration)
            return event

    @check_bucket_exists
    def _heartbeat(self, bucket_id: str, heartbeat: Event, pulsetime: float) -> Event:
        logger.debug(
            "Received heartbeat in bucket '%s'\n\ttimestamp: %s, duration: %s, pulsetime: %s\n\tdata: %s",
            bucket_id,
            heartbeat.timestamp,
            heartbeat.duration,
            pulsetime,
            heartbeat.data,
        )

        last_event = None
        if bucket_id not in self.last_event:
            last_events = self.db[bucket_id].get(limit=1)
            if len(last_events) > 0:
                last_event = last_events[0]
        else:
            last_event = copy.copy(self.last_event[bucket_id])

        if last_event:
            if last_event.data == heartbeat.data:
                merged = heartbeat_merge(last_event, heartbeat, pulsetime)
                if merged is not None:
                    # Heartbeat was merged into last_event
                    logger.debug(
                        "Received valid heartbeat, merging. (bucket: %s)", bucket_id
                    )
                    if merged.id is not None and self.db[bucket_id].replace(
                        merged.id, merged
                    ):
                        self.last_event[bucket_id] = merged
                        return merged
                    # The row may have been removed by another datastore user.
                    self.last_event.pop(bucket_id, None)
                else:
                    logger.info(
                        "Received heartbeat after pulse window, inserting as new event. (bucket: %s)",
                        bucket_id,
                    )
            else:
                logger.debug(
                    "Received heartbeat with differing data, inserting as new event. (bucket: %s)",
                    bucket_id,
                )
        else:
            logger.info(
                "Received heartbeat, but bucket was previously empty, inserting as new event. (bucket: %s)",
                bucket_id,
            )

        inserted = self.db[bucket_id].insert(heartbeat)
        self.last_event[bucket_id] = inserted
        return inserted

    def query2(self, name, query, timeperiods, cache):
        result = []
        query = "".join(query)
        for timeperiod in timeperiods:
            period = timeperiod.split("/")[
                :2
            ]  # iso8601 timeperiods are separated by a slash
            if len(period) != 2:
                raise QueryException(
                    f"Invalid timeperiod '{timeperiod}': expected two ISO8601 "
                    "datetimes separated by a slash (start/end)"
                )
            try:
                starttime = iso8601.parse_date(period[0])
                endtime = iso8601.parse_date(period[1])
            except iso8601.ParseError as e:
                raise QueryException(f"Invalid timeperiod '{timeperiod}': {e}")
            if not cache:
                result.append(query2.query(name, query, starttime, endtime, self.db))
                continue
            key = (name, query, starttime, endtime)
            hit, value, revision = self.query_cache.lookup(key)
            if not hit:
                tracker = ReadTracker(self.db)
                value = query2.query(name, query, starttime, endtime, tracker)
                self.query_cache.store(key, value, tracker, revision)
            result.append(value)
        return result

    # TODO: Right now the log format on disk has to be JSON, this is hard to read by humans...
    def get_log(self):
        """Get the server log in json format"""
        payload = []
        with open(get_log_file_path()) as log_file:
            for line in log_file.readlines()[::-1]:
                payload.append(json.loads(line))
        return payload, 200

    def get_setting(self, key):
        """Get a setting"""
        return self.settings.get(key, None)

    def set_setting(self, key, value):
        """Set a setting"""
        self.settings[key] = value
        return value
