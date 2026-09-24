import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from aw_core.models import Event
from aw_query.exceptions import QueryException

from aw_server import api as api_module
from aw_server import rest
from aw_server.query_cache import QueryCache, ReadTracker

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PERIOD = "2024-01-01T00:00:00+00:00/2024-01-02T00:00:00+00:00"
NEXT_PERIOD = "2024-01-02T00:00:00+00:00/2024-01-03T00:00:00+00:00"
QUERY = ['RETURN = query_bucket("test");']


def setup_bucket(api):
    api.create_bucket("test", "test", "test", "test")
    return api.create_events(
        "test", [Event(timestamp=START, duration=1, data={"app": "a"})]
    )[0]


def run(api, query=QUERY, period=PERIOD, cache=True):
    return api.query2("test", query, [period], cache)[0]


def spy_queries(monkeypatch):
    spy = Mock(wraps=api_module.query2.query)
    monkeypatch.setattr(api_module.query2, "query", spy)
    return spy


def test_query_cache_reuses_result_without_sharing_mutable_events(
    isolated_api, monkeypatch
):
    api = isolated_api
    setup_bucket(api)
    spy = spy_queries(monkeypatch)
    first = run(api)
    first[0].data["app"] = "changed"
    assert run(api)[0].data["app"] == "a"
    assert spy.call_count == 1
    run(api, cache=False)
    assert spy.call_count == 2
    run(api, query=["RETURN = 123;"])
    assert spy.call_count == 3


@pytest.mark.parametrize(
    "operation", ["insert", "upsert", "delete_event", "delete_bucket", "metadata"]
)
def test_query_cache_invalidates_mutations(isolated_api, monkeypatch, operation):
    api = isolated_api
    event = setup_bucket(api)
    spy = spy_queries(monkeypatch)
    run(api)
    if operation == "insert":
        api.create_events(
            "test", [Event(timestamp=START + timedelta(seconds=2), data={"app": "b"})]
        )
    elif operation == "upsert":
        # Bulk upserts can move events out of a cached period.
        api.create_events(
            "test",
            [
                Event(
                    id=event.id, timestamp=START + timedelta(days=2), data={"app": "b"}
                ),
                Event(timestamp=START + timedelta(days=3)),
            ],
        )
    elif operation == "delete_event":
        api.delete_event("test", event.id)
    elif operation == "delete_bucket":
        api.delete_bucket("test")
        api.create_bucket("test", "test", "test", "test")
    else:
        api.update_bucket("test", data={"label": "new"})
    assert run(api) == run(api, cache=False)
    assert spy.call_count == 3


def test_heartbeat_preserves_unaffected_days_and_invalidates_current_day(
    isolated_api, monkeypatch
):
    api = isolated_api
    setup_bucket(api)
    current = START + timedelta(days=1, hours=1)
    api.heartbeat("test", Event(timestamp=current, data={"app": "live"}), 60)
    spy = spy_queries(monkeypatch)
    run(api)
    run(api, period=NEXT_PERIOD)
    api.heartbeat(
        "test",
        Event(timestamp=current + timedelta(seconds=10), data={"app": "live"}),
        60,
    )
    run(api)
    assert spy.call_count == 2
    assert run(api, period=NEXT_PERIOD)[0].duration == timedelta(seconds=10)
    assert spy.call_count == 3


def test_cache_tracks_overridden_query_time_bounds(isolated_api, monkeypatch):
    api = isolated_api
    setup_bucket(api)
    query = [
        'STARTTIME = "2024-01-01T00:00:00+00:00"; ENDTIME = "2024-01-02T00:00:00+00:00"; RETURN = query_bucket("test");'
    ]
    spy = spy_queries(monkeypatch)
    run(api, query=query, period=NEXT_PERIOD)
    api.heartbeat(
        "test", Event(timestamp=START + timedelta(seconds=10), data={"app": "a"}), 60
    )
    assert run(api, query=query, period=NEXT_PERIOD)[0].duration == timedelta(
        seconds=10
    )
    assert spy.call_count == 2


def test_cache_invalidates_bucket_discovery_and_import(isolated_api, monkeypatch):
    api = isolated_api
    setup_bucket(api)
    query = ['RETURN = find_bucket("test");']
    spy = spy_queries(monkeypatch)
    assert run(api, query) == "test"
    api.delete_bucket("test")
    with pytest.raises(QueryException):
        run(api, query)
    api.import_bucket(
        {
            "id": "test-import",
            "type": "test",
            "client": "test",
            "hostname": "test",
            "created": START.isoformat(),
            "events": [],
        }
    )
    assert run(api, query) == "test-import"
    assert spy.call_count == 3


def test_query_overlapping_write_is_not_stored(isolated_api, monkeypatch):
    api = isolated_api
    setup_bucket(api)
    original = api_module.query2.query
    calls = []

    def evaluate(*args):
        value = original(*args)
        calls.append(True)
        if len(calls) == 1:
            api.delete_event("test", value[0].id)
        return value

    monkeypatch.setattr(api_module.query2, "query", evaluate)
    assert len(run(api)) == 1
    assert run(api) == []
    assert len(calls) == 2


def test_query_cache_bounds_expiry_and_active_writers(monkeypatch):
    cache = QueryCache(max_entries=2, max_bytes=4096, ttl=10)
    tracker = ReadTracker(None)
    monkeypatch.setattr("aw_server.query_cache.monotonic", lambda: 0)
    for key in ("a", "b", "c"):
        cache.store(key, [1], tracker, cache.lookup(key)[2])
    assert not cache.lookup("a")[0]
    assert cache.lookup("b")[0]
    assert cache.bytes <= 4096
    cache.store("large", "x" * 5000, tracker, cache.lookup("large")[2])
    assert not cache.lookup("large")[0]
    with cache.mutation("test"):
        assert not cache.lookup("b")[0]
        cache.store("during_write", [1], tracker, cache.lookup("x")[2])
    assert not cache.lookup("during_write")[0]
    monkeypatch.setattr("aw_server.query_cache.monotonic", lambda: 11)
    assert not cache.lookup("b")[0]
    assert cache.bytes == 0


def test_heartbeat_uses_inserted_id_and_handles_intervening_mutations(isolated_api):
    api = isolated_api
    api.create_bucket("test", "test", "test", "test")
    first = api.heartbeat("test", Event(timestamp=START, data={"app": "a"}), 60)
    assert first.id is not None
    second = api.heartbeat(
        "test", Event(timestamp=START + timedelta(seconds=1), data={"app": "a"}), 60
    )
    assert second.id == first.id
    api.delete_event("test", first.id)
    replacement = api.heartbeat(
        "test", Event(timestamp=START + timedelta(seconds=2), data={"app": "a"}), 60
    )
    assert api.get_eventcount("test") == 1
    assert replacement.duration == timedelta(0)
    api.delete_bucket("test")
    api.create_bucket("test", "test", "test", "test")
    api.heartbeat(
        "test", Event(timestamp=START + timedelta(seconds=3), data={"app": "a"}), 60
    )
    assert api.get_eventcount("test") == 1


def test_failed_heartbeat_does_not_advance_cached_duration(isolated_api, monkeypatch):
    api = isolated_api
    setup_bucket(api)
    api.heartbeat(
        "test", Event(timestamp=START + timedelta(seconds=1), data={"app": "a"}), 60
    )
    before = api.last_event["test"].duration
    monkeypatch.setattr(
        api.db["test"], "replace", Mock(side_effect=RuntimeError("write failed"))
    )
    with pytest.raises(RuntimeError, match="write failed"):
        api.heartbeat(
            "test", Event(timestamp=START + timedelta(seconds=5), data={"app": "a"}), 60
        )
    assert api.last_event["test"].duration == before


def test_heartbeat_serializes_direct_api_calls(isolated_api):
    api = isolated_api
    if api.db.storage_strategy.sid != "memory":
        pytest.skip(
            "Memory backend isolates API synchronization from DB thread support"
        )
    api.create_bucket("test", "test", "test", "test")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: api.heartbeat(
                    "test", Event(timestamp=START, data={"app": "a"}), 60
                ),
                range(50),
            )
        )
    assert api.get_eventcount("test") == 1


def test_stream_export_matches_eager_export_and_is_lazy(isolated_api, monkeypatch):
    api = isolated_api
    setup_bucket(api)
    api.create_bucket("empty", "test", "test", "test")
    expected = {"buckets": api.export_all()}
    assert json.loads("".join(api.stream_export())) == expected
    assert json.loads("".join(api.stream_export("test"))) == {
        "buckets": {"test": expected["buckets"]["test"]}
    }
    monkeypatch.setattr(
        api.db.storage_strategy,
        "get_events",
        Mock(side_effect=AssertionError("eager read")),
    )
    # None of the built-in iterators should fall back to materializing get_events.
    assert json.loads("".join(api.stream_export())) == expected


def test_closing_export_closes_event_iterator(isolated_api, monkeypatch):
    api = isolated_api
    setup_bucket(api)
    consumed, closed = [], []

    def events():
        try:
            for _ in range(10000):
                consumed.append(True)
                yield Event(timestamp=START)
        finally:
            closed.append(True)

    monkeypatch.setattr(api.db["test"], "iter_events", events)
    stream = api.stream_export("test")
    while not consumed:
        next(stream)
    stream.close()
    assert 0 < len(consumed) < 10000
    assert closed == [True]


def test_http_stream_export_and_cache_options(flask_client, app):
    response = flask_client.get("/api/0/export")
    assert response.status_code == 200
    assert response.is_streamed
    assert response.mimetype == "application/json"
    assert "buckets" in response.json
    assert flask_client.get("/api/0/buckets/missing/export").status_code == 404
    body = {"query": ["RETURN = 1;"], "timeperiods": [PERIOD]}
    assert flask_client.post("/api/0/query/?cache=0", json=body).json == [1]
    assert (
        flask_client.post("/api/0/query/?cache=invalid", json=body).status_code == 400
    )


def test_disabled_debug_logging_does_not_format_payload(isolated_api, monkeypatch, app):
    class Payload(dict):
        def __str__(self):
            raise AssertionError("payload was eagerly formatted")

    api = isolated_api
    api.create_bucket("test", "test", "test", "test")
    monkeypatch.setattr(api_module.logger, "level", logging.INFO)
    api.heartbeat("test", Event(timestamp=START, data=Payload(app="a")), 60)
    monkeypatch.setattr(rest.logger, "level", logging.INFO)
    with app.test_request_context("/api/0/buckets/test/events", method="POST"):
        monkeypatch.setattr(
            rest.request._get_current_object(),
            "get_json",
            lambda: Payload(timestamp=START, data={"app": "a"}),
        )
        monkeypatch.setattr(app.api, "create_events", lambda *args: [])
        assert rest.EventsResource().post("test") == ([], 200)
