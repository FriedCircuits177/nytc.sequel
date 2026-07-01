import threading


def construct_thread(fn, *args, daemon=True, **kwargs) -> threading.Thread:
    # Safely extract the class name if it's a bound method, otherwise default to empty string
    cls = f"{fn.__self__.__class__.__name__}." if hasattr(fn, "__self__") else ""

    return threading.Thread(
        target=fn,
        args=args,
        kwargs=kwargs,
        name=f"{fn.__module__}.{cls}{fn.__name__}",
        daemon=daemon,
    )
