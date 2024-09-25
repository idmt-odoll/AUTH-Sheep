try:
    from .BoT_SORT import BOTSORT
    from .ByteTracker import BYTETracker
except ImportError or ModuleNotFoundError:
    from BoT_SORT import BOTSORT
    from ByteTracker import BYTETracker

available_trackers = ["bytetrack", "botsort"]

tracker_dict = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT
}

