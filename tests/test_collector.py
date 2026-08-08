from cua_jepa.collector import BundleCollector


def test_url_with_sid_replaces_existing_session_and_keeps_route() -> None:
    assert BundleCollector._url_with_sid(
        "http://127.0.0.1:5173/calendar/week?view=work&sid=old#today", "new"
    ) == "http://127.0.0.1:5173/calendar/week?view=work&sid=new#today"
